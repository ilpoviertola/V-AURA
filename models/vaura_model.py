"""
Adapted from: https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/models/lm.py
"""

from pathlib import Path
from typing import Union, List, Tuple, Optional, Any, Dict

import torch
import pytorch_lightning as pl

from utils.utils import (
    instantiate_from_config,
    get_filename_and_parent_dir_from_path,
    sample_top_k,
    sample_top_p,
    multinomial,
)
from utils.train_utils import (
    disabled_train,
    generate_video_from_attn_weights,
    combine_attn_weights_to_tensor,
)
from utils.data_utils import scale_tensor
from models.modules.misc.codebook_patterns import DelayedPatternProvider


class VAURAModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 5e-6,
        lr_scheduler: dict = None,
        weight_decay: float = 0.01,
        betas: tuple = (0.9, 0.95),
        batch_size: int = 1,
        use_visual_conditioning: bool = True,
        feature_extractor_config: dict = None,
        audio_encoder_config: dict = None,
        sampler_config: dict = None,
        visual_bridge_config: dict = None,
        pattern_provider_config: dict = None,
        predict_at_val_start: bool = False,
        return_attention_weights: bool = False,
        plot_distr_of_pred_indices: bool = False,
        freeze_feature_extractor: bool = False,
        files_to_track_during_training: List[str] = None,
        flatten_vis_feats: bool = False,
        apply_per_video_frame_mask: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Hyperparameters for optimizer
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.batch_size = batch_size
        self.lr_scheduler_cfg = lr_scheduler
        self.lr_scheduler = None

        # Model components
        # FIXME: Audio and video fe ckpts are saved to the model ckpt
        self.use_visual_conditioning = use_visual_conditioning
        self.freeze_feature_extractor = freeze_feature_extractor
        self.visual_feature_extractor = (
            instantiate_from_config(feature_extractor_config)
            if self.use_visual_conditioning
            else None
        )
        if freeze_feature_extractor:
            self.visual_feature_extractor.eval()
            self.visual_feature_extractor.requires_grad_(False)
            self.visual_feature_extractor.train = disabled_train
        self.using_avclip = (
            self.visual_feature_extractor.__class__.__name__ == "MotionFormer"
        )
        self.flatten_vis_feats = self.using_avclip and flatten_vis_feats
        self.sampler = instantiate_from_config(
            self._update_sampler_config(sampler_config)
        )
        self.visual_bridge = (
            instantiate_from_config(visual_bridge_config)
            if self.use_visual_conditioning
            else None
        )
        self.audio_encoder = instantiate_from_config(audio_encoder_config)
        if hasattr(self.sampler, "initialize_embeddings"):
            if self.audio_encoder.__class__.__name__ == "DacModelWrapper":
                self.sampler.initialize_embeddings(self.audio_encoder.model)
        self.audio_encoder.eval()
        self.audio_encoder.requires_grad_(False)
        self.audio_encoder.train = disabled_train
        self.audio_encoder.model.half()
        self.num_codebooks = self.sampler.num_codebooks
        if pattern_provider_config is not None:
            self.pattern_provider = instantiate_from_config(
                self._check_codebook_pattern_config(pattern_provider_config)
            )
        else:
            self.pattern_provider = DelayedPatternProvider(n_q=self.num_codebooks)
        if hasattr(self.sampler, "codebook_pattern"):
            self.sampler.codebook_pattern = self.pattern_provider.__class__.__name__
        self.pattern = None
        self.apply_per_video_frame_mask = apply_per_video_frame_mask

        # Loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Logging related attributes
        self.first_val_epoch = True
        self.frames_logged = False
        self.predict_at_val_start = predict_at_val_start
        self.return_attention_weights = return_attention_weights
        self.plot_distr_of_pred_indices = plot_distr_of_pred_indices
        self.a_sr = None
        self.v_sr = None
        self.files_to_track_during_training = (
            files_to_track_during_training
            if files_to_track_during_training != None
            else []
        )

        # NOTE: Check reading checkpoint here instead of trainer?

    @property
    def special_token_id(self) -> int:
        """This is codebook padding + BOS token."""
        return self.sampler.d_codebook

    @property
    def max_duration(self, max_duration: float = 0.64) -> float:
        if hasattr(self, "_trainer") and self._trainer is not None:
            return self.trainer.val_dataloaders.dataset.video_length
        else:
            return max_duration

    def forward(
        self,
        frames: torch.Tensor,
        audio: torch.Tensor,
        clip_indices: Optional[torch.Tensor] = None,
    ):
        """Run V-AURA model.

        Args:
            frames (torch.Tensor): RGB frames (B, C, Tv, H, W)
            audio (torch.Tensor): Mono audio (B, 1, Ta)
            clip_indices (torch.Tensor): Indices of clips in the batch (B,)

        Returns:
            tuple: logits and ground truth codebook indices
        """
        aud_feats = self.audio_encoder.encode(audio)
        B, _, Ta = aud_feats.shape
        vis_feats = self._handle_visual_conditioning(frames, clip_indices, B)

        # map codes [B, Da, Ta] into pattern sequence [B, Da, Sa] using special_token_id for masked tokens
        if self.pattern is None:
            self.pattern = self.pattern_provider.get_pattern(Ta)

        # Builds sequence by prepending special tokens (BOS-tokens) to sequence (and possibly to other places).
        # This is why no BOS tokens are preprended (explicitly). To shift tokens (aud_feats) one step
        # to the right.
        sequence_codes, _, _ = self.pattern.build_pattern_sequence(
            aud_feats[:, : self.num_codebooks, :-1].contiguous(),
            self.special_token_id,
            keep_only_valid_steps=False,
        )
        # TODO: Think about reshaping the sequences back to original shapes
        # (concatenate the clips of differnet views)
        logits, _, _ = self.sampler(
            tgt=sequence_codes,
            memory=vis_feats,
            tgt_is_causal=True,
            use_conditioning=self.use_visual_conditioning,
            return_attention_weights=False,
            apply_per_video_frame_mask=self.apply_per_video_frame_mask,
        )
        # map back the logits on pattern sequence to logits on original codes: [B, Da, Sa, card] -> [B, Da, Ta, card]
        # and provide the corresponding mask over invalid positions of tokens
        logits = logits.permute(0, 3, 1, 2)  # [B, card, Da, Sa]
        # if visual conditioning was prepended, remove it
        # otherwise this does nothing
        logits = logits[..., -sequence_codes.shape[-1] :]
        # note: we use nans as special token to make it obvious if we feed unexpected logits
        logits, _, logits_mask = self.pattern.revert_pattern_logits(
            logits, float("nan"), keep_only_valid_steps=False
        )
        logits = logits.permute(0, 2, 3, 1)  # [B, Da, Ta, card]
        logits_mask = logits_mask[None, :, :].expand(
            B, -1, -1
        )  # [Da, Ta] -> [B, Da, Ta]
        return logits, logits_mask, aud_feats

    def _handle_visual_conditioning(
        self, frames: torch.Tensor, clip_indices: torch.Tensor, B: int
    ):
        if self.use_visual_conditioning:
            assert frames is not None
            if self.using_avclip:
                vis_feats, _ = self.visual_feature_extractor(frames)
                # concat clips
                if self.flatten_vis_feats:  # num_clips
                    B, S, Tv, D = vis_feats.shape
                    vis_feats = vis_feats.reshape(B, S * Tv, D)
            else:
                vis_feats = self.visual_feature_extractor(
                    frames
                )  # [B, Dv, Tv, H, W] (for S3D: Dv=1024, H=W=7)
            if self.freeze_feature_extractor:
                vis_feats = vis_feats.detach()
            vis_feats = self.visual_bridge(vis_feats)
        else:
            vis_feats = None
        return vis_feats

    def _flatten_vis_feats(
        self, vis_feats: torch.Tensor, B: int, S: int, Tv: int, D: int
    ) -> torch.Tensor:
        vis_feats_flattened = torch.empty(
            B, Tv * S, D, dtype=vis_feats.dtype, device=vis_feats.device
        )
        for i in range(B):
            for j in range(S):
                vis_feats_flattened[i, Tv * j : Tv * (j + 1), :] = vis_feats[
                    i + (j * B)
                ]
        return vis_feats_flattened

    def _stack_list_repr(
        self, list_repr: List[List[torch.Tensor]], to_3dim: bool = False
    ) -> torch.Tensor:
        """Stack a list of lists of tensors into a single tensor.

        Args:
            list_repr (List[List[torch.Tensor]]): List of lists of tensors (num_clips, num_views, B, ...).
            to_3dim (bool): Whether to stack to 3D tensor.

        Returns:
            torch.Tensor: Stacked tensor.
        """
        tensor_repr = torch.stack([torch.stack(tensors) for tensors in list_repr])
        if to_3dim:
            tensor_repr = tensor_repr.view(-1, *tensor_repr.shape[-2:])
        return tensor_repr

    def _compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Compute cross entropy between multi-codebook targets and model's logits.
        The cross entropy is computed per codebook to provide codebook-level cross entropy.
        Valid timesteps for each of the codebook are pulled from the mask, where invalid
        timesteps are set to 0.

        Args:
            logits (torch.Tensor): Model's logits of shape [B, K, T, card].
            targets (torch.Tensor): Target codes, of shape [B, K, T].
            mask (torch.Tensor): Mask for valid target codes, of shape [B, K, T].
        Returns:
            loss (torch.Tensor): Cross entropy averaged over the codebooks
            loss_per_codebook (list of torch.Tensor): Cross entropy per codebook (detached).
        """
        B, K, T = targets.shape
        assert logits.shape[:-1] == targets.shape
        assert mask.shape == targets.shape
        loss = torch.zeros([], device=targets.device)
        loss_per_codebook: List[torch.Tensor] = []
        for k in range(K):
            logits_k = (
                logits[:, k, ...].contiguous().view(-1, logits.size(-1))
            )  # [B x T, card]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
            ce_targets = targets_k[mask_k]
            ce_logits = logits_k[mask_k]
            q_ce = self.loss_fn(ce_logits, ce_targets)
            loss += q_ce
            loss_per_codebook.append(q_ce.detach())
        # average cross entropy across codebooks
        loss = loss / K
        return loss, loss_per_codebook

    def _shared_step(self, batch, batch_idx):
        audio = (
            batch["audio"]
            if self.flatten_vis_feats
            else self._stack_list_repr(batch["audio"], to_3dim=True)
        )  # .unsqueeze(1)
        frames = batch["frames"]  # .permute(0, 2, 1, 3, 4)
        logits, logits_mask, target = self.forward(
            frames, audio, batch["meta"].get("clip_indices", None)
        )
        loss, loss_per_cb = self._compute_loss(
            logits, target[:, : self.num_codebooks, :], logits_mask
        )
        return logits, target, loss, loss_per_cb

    def _shared_log(self, stage: str, loss: torch.Tensor, loss_per_cb: torch.Tensor):
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log_loss_per_codebook(
            f"{stage}_loss_per_codebook",
            loss_per_cb,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

    def training_step(self, batch, batch_idx):
        logits, _, loss, loss_per_cb = self._shared_step(batch, batch_idx)
        files_to_track = list(
            filter(
                lambda x: Path(x).stem in self.files_to_track_during_training,
                batch["meta"]["filepath"],
            )
        )
        if self.training and files_to_track:
            indices_of_common_elements = [
                batch["meta"]["filepath"].index(fn) for fn in files_to_track
            ]
            self._log_training_samples(
                logits.detach().clone(),
                files_to_track,
                indices_of_common_elements,
            )
        self._shared_log("train", loss, loss_per_cb)
        return loss

    def validation_step(self, batch, batch_idx):
        _, _, loss, loss_per_cb = self._shared_step(batch, batch_idx)
        self._shared_log("val", loss, loss_per_cb)
        return loss

    def test_step(self, batch, batch_idx):
        _, _, loss, loss_per_cb = self._shared_step(batch, batch_idx)
        self._shared_log("test", loss, loss_per_cb)
        return loss

    def on_validation_epoch_start(self) -> None:
        if not self.predict_at_val_start or self.trainer.sanity_checking:
            return

        # Only log the conditioned frames once
        if not self.frames_logged and self.use_visual_conditioning:
            for _, batch in enumerate(self.trainer.datamodule.predict_dataloader()):
                # Set some attributes for logging
                if self.first_val_epoch:
                    self.a_sr = batch["meta"]["audio_fps"][0].item()
                    self.v_sr = batch["meta"]["video_fps"][0].item()
                    self.first_val_epoch = False
                item = self.transfer_batch_to_device(
                    batch, self.device, dataloader_idx=0
                )
                fn = get_filename_and_parent_dir_from_path(item["meta"]["filepath"][0])
                frames = item["frames"]
                B, num_clips, C, Tv, H, W = frames.shape
                assert B == 1, "batch size and amount of spatial views must be 1"
                # rescale tensor to make colors look better for human viewers
                # will not result to original but still better...
                frames = scale_tensor(frames[0, ...].permute(0, 2, 1, 3, 4), 0, 1)
                video = frames.reshape(1, Tv * num_clips, C, H, W)
                self.logger.experiment.add_video(
                    f"conditioned_frames/{fn}",
                    video,
                    self.global_step,
                    fps=self.v_sr,
                )
                self._log_predict_run(item)
            self.frames_logged = True
            return

        # Log the inference runs
        for _, batch in enumerate(self.trainer.datamodule.predict_dataloader()):
            item = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
            self._log_predict_run(item)
            # Only log one inference run if no conditioning is used
            if not self.use_visual_conditioning:
                return

    def configure_optimizers(self):
        sampler_optim = self._configure_sampler_optimizers(
            weight_decay=self.weight_decay,
            learning_rate=self.lr,
            betas=self.betas,
            device_type=self.device.type,
        )
        if self.lr_scheduler_cfg is not None:
            self.lr_scheduler_cfg["params"]["optimizer"] = sampler_optim
            lr_scheduler = instantiate_from_config(self.lr_scheduler_cfg)
            return {
                "optimizer": sampler_optim,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return sampler_optim

    @torch.no_grad()
    def generate(
        self,
        frames: Union[List[List[torch.Tensor]], None] = None,
        audio: Union[torch.Tensor, None] = None,
        clip_indices: Union[torch.Tensor, None] = None,
        max_new_tokens: int = 512,
        return_attention_weights: bool = False,
        return_sampled_indices: bool = False,
        check: bool = False,
        use_sampling: bool = True,
        temp: float = 1.0,
        top_k: int = 256,
        top_p: float = 0.0,
        remove_prompts: bool = False,
        prompt_is_encoded: bool = False,
        cfg_scale: float = 1.0,
    ) -> dict:
        """Run V-AURA model.

        Args:
            frames (torch.Tensor): RGB frames (B, C, Tv, H, W) (required).
            audio (torch.Tensor): Mono audio (B, 1, Ta) (optional).

        Returns:
            dict: Generated audio, attention weights and sampled indices.
        """
        assert not self.training, "do not use generation in training mode"
        generated_item = {}

        possible_num_samples = []
        if frames is not None:
            if self.flatten_vis_feats:
                possible_num_samples.append(frames.shape[0])
            else:
                possible_num_samples.append(frames.shape[0])
        elif audio is not None:
            possible_num_samples.append(audio.shape[0])
        else:
            possible_num_samples.append(1)
        assert [
            x == possible_num_samples[0] for x in possible_num_samples
        ], "Inconsistent inputs shapes"
        num_samples = possible_num_samples[0]

        if audio is None:
            assert num_samples > 0
            audio = torch.zeros(
                (num_samples, self.num_codebooks, 0),
                dtype=torch.long,
                device=self.device,
            )
        else:
            if not prompt_is_encoded:
                audio = self.audio_encoder.encode(audio)
                audio = (
                    torch.cat([encoded[0] for encoded in audio], dim=-1)
                    .to(self.device)
                    .detach()
                )  # [B, Da, Ta]

        B, K, T = audio.shape
        vis_feats = self._handle_visual_conditioning(
            frames, clip_indices, B
        )  # [B, Dv, Tv, H, W] (for S3D: Dv=1024, H=W=7)
        start_offset = T
        assert (
            start_offset < max_new_tokens
        ), "gt audio prompt can not be longer than max_new_tokens"

        pattern = self.pattern_provider.get_pattern(max_new_tokens)
        # this token is used as default value for codes that are not generated yet
        unknown_token = -1

        # we generate codes up to the max_new_tokens that will be mapped to the pattern sequence
        gen_codes = torch.full(
            (B, K, max_new_tokens), unknown_token, dtype=torch.long, device=self.device
        )
        # filling the gen_codes with the prompt if needed
        gen_codes[..., :start_offset] = audio
        # create the gen_sequence with proper interleaving from the pattern: [B, K, S]
        gen_sequence, indexes, mask = pattern.build_pattern_sequence(
            gen_codes, self.special_token_id
        )
        # retrieve the start_offset in the sequence:
        # it is the first sequence step that contains the `start_offset` timestep
        start_offset_sequence = pattern.get_first_step_with_timesteps(start_offset)
        assert start_offset_sequence is not None

        prev_offset = 0
        gen_sequence_len = gen_sequence.shape[-1]  # gen_sequence shape is [B, K, S]
        attention_weights = {"mha_w": [], "sa_w": []}
        for offset in range(start_offset_sequence, gen_sequence_len):
            # get current sequence
            # have to feed the model with the whole sequence everytime since
            # no caching is implemented :(
            curr_sequence = gen_sequence[..., prev_offset:offset]
            curr_mask = mask[None, ..., prev_offset:offset].expand(B, -1, -1)
            if check:
                # check coherence between mask and sequence
                assert (
                    curr_sequence
                    == torch.where(curr_mask, curr_sequence, self.special_token_id)
                ).all()
                # should never happen as gen_sequence is filled progressively
                assert not (curr_sequence == unknown_token).any()
            # sample next token from the model, next token shape is [B, K, 1]
            next_token, sa_w, mha_w = self._sample_next_token(
                curr_sequence,
                vis_feats.detach().clone() if self.use_visual_conditioning else None,
                use_sampling,
                temp,
                top_k,
                top_p,
                return_attention_weights,
                cfg_scale,
            )
            if return_attention_weights:
                attention_weights["sa_w"].append(sa_w[-1, -1, :].cpu())
                if (
                    self.use_visual_conditioning
                    and self.sampler.__class__.__name__ != "ChannelFeatConcatSampler"
                ):  # when no conditioning is used this block is omitted
                    attention_weights["mha_w"].append(mha_w[-1, -1, :].cpu())
            # ensure the tokens that should be masked are properly set to special_token_id
            # as the model never output special_token_id
            valid_mask = mask[..., offset : offset + 1].expand(B, -1, -1)
            next_token[~valid_mask] = self.special_token_id
            # ensure we don't overwrite prompt tokens, we only write over unknown tokens
            # (then mask tokens should be left as is as well, which is correct)
            gen_sequence[..., offset : offset + 1] = torch.where(
                gen_sequence[..., offset : offset + 1] == unknown_token,
                next_token,
                gen_sequence[..., offset : offset + 1],
            )
            # have to feed the model with the whole sequence everytime since
            # no caching is implemented :(
            # prev_offset = offset

        # ensure sequence has been entirely filled
        assert not (gen_sequence == unknown_token).any()
        # ensure gen_sequence pattern and mask are matching
        # which means the gen_sequence is valid according to the pattern
        assert (
            gen_sequence
            == torch.where(
                mask[None, ...].expand(B, -1, -1), gen_sequence, self.special_token_id
            )
        ).all()
        # get back the codes, trimming the prompt if needed and cutting potentially incomplete timesteps
        out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(
            gen_sequence, special_token=unknown_token
        )

        # sanity checks over the returned codes and corresponding masks
        assert (out_codes[..., :max_new_tokens] != unknown_token).all()
        assert (out_mask[..., :max_new_tokens] == 1).all()

        out_start_offset = start_offset if remove_prompts else 0
        out_codes = out_codes[..., out_start_offset:max_new_tokens]

        # ensure the returned codes are all valid
        assert (out_codes >= 0).all() and (out_codes <= self.sampler.d_codebook).all()

        # gather outputs
        sampled_frames = [(out_codes[..., : self.num_codebooks, :], None)]
        generated_audio = self.audio_encoder.decode(sampled_frames)
        generated_item["generated_audio"] = generated_audio
        s_attn_weights = (
            combine_attn_weights_to_tensor(attention_weights["sa_w"])
            if return_attention_weights
            else None
        )
        mh_attn_weights = (
            combine_attn_weights_to_tensor(attention_weights["mha_w"])
            if return_attention_weights
            and (
                self.use_visual_conditioning
                and self.sampler.__class__.__name__ != "ChannelFeatConcatSampler"
            )
            else None
        )
        generated_item["s_attn_weights"] = s_attn_weights
        generated_item["mha_attn_weights"] = mh_attn_weights
        generated_item["sampled_indices"] = (
            out_codes if return_sampled_indices else None
        )
        return generated_item

    def _configure_sampler_optimizers(
        self, weight_decay, learning_rate, betas, device_type
    ):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

        return optimizer

    def _log_training_samples(
        self,
        logits: torch.Tensor,
        filenames: List[str],
        indices: Union[List[int], None] = None,
    ):
        """Log training samples using greedy sampling for debugging purposes."""
        tokens = torch.argmax(logits[indices, ...], dim=-1, keepdim=False)
        sampled_frames = [(tokens[..., : self.num_codebooks, :], None)]
        generated_audios = self.audio_encoder.decode(sampled_frames)
        generated_audios = torch.clamp(generated_audios, -1.0, 1.0)
        assert self.logger is not None
        for fn, generated_audio in zip(filenames, generated_audios):
            self.logger.experiment.add_audio(
                f"generated_audio_of_training_data/{fn}",
                generated_audio.squeeze(0),
                self.global_step,
                self.a_sr or 24000,
            )

    def _log_predict_run(self, item: dict):
        fn = get_filename_and_parent_dir_from_path(item["meta"]["filepath"][0])
        frames = item["frames"]  # .permute(0, 2, 1, 3, 4)
        clip_indices = item["meta"].get("clip_indices", None)

        # TODO: Deduct max_new_tokens from the length of the audio
        generated_item = self.generate(
            frames,
            clip_indices=clip_indices,
            max_new_tokens=(
                221 if self.flatten_vis_feats or not self.using_avclip else 48
            ),
            return_attention_weights=self.return_attention_weights,
            return_sampled_indices=self.plot_distr_of_pred_indices,
        )
        generated_audio = generated_item["generated_audio"]
        s_attn_weights = generated_item["s_attn_weights"]
        mh_attn_weights = generated_item["mha_attn_weights"]
        sampled_indices = generated_item["sampled_indices"]

        assert self.logger is not None
        # Scale audio to [-1, 1]
        generated_audio = torch.clamp(generated_audio, -1.0, 1.0)
        self.logger.experiment.add_audio(
            f"generated_audio/{fn}",
            generated_audio.reshape(1, -1).squeeze(0),
            self.global_step,
            self.a_sr,
        )

        if self.return_attention_weights:
            self.logger.experiment.add_video(
                f"s_attention_weights/{fn}",
                generate_video_from_attn_weights(s_attn_weights, (50, 5)),
                self.global_step,
                fps=self.v_sr,
            )
            if mh_attn_weights is not None:  # Is None when features are prepended
                self.logger.experiment.add_video(
                    f"mh_attention_weights/{fn}",
                    generate_video_from_attn_weights(mh_attn_weights, (50, 5)),
                    self.global_step,
                    fps=self.v_sr,
                )

        if self.plot_distr_of_pred_indices:
            self.logger.experiment.add_histogram(
                f"sampled_indices/{fn}",
                sampled_indices.squeeze(0),
                self.global_step,
            )

    def _update_sampler_config(self, cfg: dict) -> dict:
        """
        Add some necessary attributes to the sampler config.
        Can not be done with OmegaConf's resolve since sampler
        and main config are physically different files.
        """
        cfg["params"]["use_visual_conditioning"] = self.use_visual_conditioning
        return cfg

    def _check_codebook_pattern_config(self, cfg: dict) -> dict:
        """
        Double check that the number of codebooks is set correctly.

        Args:
            cfg (dict): Config to check

        Returns:
            dict: Possibly updated config
        """
        if cfg["params"]["n_q"] != self.num_codebooks:
            self.print(
                "WARNING: Changing n_q in codebook pattern config to match model"
            )
            cfg["params"]["n_q"] = self.num_codebooks
        return cfg

    def log_loss_per_codebook(
        self,
        name: str,
        loss_per_cb: List[torch.Tensor],
        batch_size: int,
        prog_bar: bool = True,
        on_step: bool = False,
        on_epoch: bool = True,
        logger: bool = True,
        sync_dist: bool = True,
    ):
        for i, loss in enumerate(loss_per_cb):
            self.log(
                f"{name}_{i}",
                loss,
                prog_bar=prog_bar,
                on_step=on_step,
                on_epoch=on_epoch,
                logger=logger,
                sync_dist=sync_dist,
                batch_size=batch_size,
            )

    def on_fit_start(self) -> None:
        # Group the codebook losses together
        assert self.logger is not None
        tb = self.logger.experiment
        layout: Dict[str, Any] = {}
        # (metric group, display name, [plot type, [metric names (must match to ones logged)]])
        grouped_metrics = [
            (
                "metrics",
                "train_loss_per_codebook",
                [
                    "Multiline",
                    [
                        f"train_loss_per_codebook_{i}"
                        for i in range(self.sampler.num_codebooks)
                    ],
                ],
            ),
            (
                "metrics",
                "val_loss_per_codebook",
                [
                    "Multiline",
                    [
                        f"val_loss_per_codebook_{i}"
                        for i in range(self.sampler.num_codebooks)
                    ],
                ],
            ),
        ]
        for group in grouped_metrics:
            if group[0] not in layout:
                layout[group[0]] = {}
            layout[group[0]][group[1]] = group[2]
        tb.add_custom_scalars(layout)

    def _sample_next_token(
        self,
        sequence: torch.Tensor,
        condition: torch.Tensor,
        use_sampling: bool = False,
        temp: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        return_attention_weights: bool = False,
        cfg_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, Any, Any]:
        use_cfg = (
            cfg_scale > 1.0 and self.sampler.__class__.__name__ == "Transformer"
        )  # Llama-like model class name
        if use_cfg:
            cond_null = (
                torch.zeros_like(condition)
                + self.sampler.cls_embeddings.uncond_embedding
            )
            condition = torch.cat([condition, cond_null], dim=0)
            sequence = sequence.repeat(2, 1, 1)

        with torch.no_grad():
            logits, s_attn_weights, mh_attn_weights = self.sampler(
                tgt=sequence,
                memory=condition,
                use_conditioning=(condition is not None),
                tgt_is_causal=True,
                return_attention_weights=return_attention_weights,
                apply_per_video_frame_mask=self.apply_per_video_frame_mask,
            )

        logits = logits.permute(0, 1, 3, 2)  # [B, K, card, T]
        logits = logits[..., -1]  # [B x K x card]

        if use_cfg:
            cond_logits = logits[: condition.size(0) // 2]
            uncond_logits = logits[condition.size(0) // 2 :]
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale

        # Apply softmax for sampling if temp > 0. Else, do greedy sampling to avoid zero division error.
        if use_sampling and temp > 0.0:
            probs = torch.softmax(logits / temp, dim=-1)
            if top_p > 0.0:
                next_token = sample_top_p(probs, p=top_p)
            elif top_k > 0:
                next_token = sample_top_k(probs, k=top_k)
            else:
                next_token = multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        return next_token, s_attn_weights, mh_attn_weights
