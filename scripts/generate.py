"""Inference script for the model. Use through main.py."""

from pathlib import Path
import traceback
import logging
import typing as tp
from shutil import copyfile
import multiprocessing
import warnings
from math import ceil

import torchaudio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

from models.vaura_model import VAURAModel
from models.modules.sampler.llama import Transformer
from utils.utils import get_file_with_best_val_loss, write_video
from utils.data_utils import read_video_to_frames_and_audio_streams, normalize_audio
from utils.train_utils import get_datamodule_from_type
from models.data.transforms.video_transforms import (
    get_resize_and_convert_to_float32_transforms,
)
from models.data.vggsound_dataset import EPS as EPS_VGGSOUND
from models.data.greatesthit_dataset import EPS as EPS_GREATESTHIT


COMPRESSION_MODEL_FRAME_RATE = 86
DEFAULT_OVERWRITE_HPARAMS = {
    "feature_extractor_config": {
        "params": {"ckpt_path": "./segment_avclip/vggsound/best.pt"}
    }
}
logger = logging.getLogger(__name__)
logging_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(logging_format))
logger.addHandler(stream_handler)


def resolve_ckpt(cfg: DictConfig) -> Path:
    assert "checkpoint_path" in cfg, "Checkpoint path must be defined in config."
    checkpoint_path = Path(cfg.checkpoint_path)
    assert (
        checkpoint_path.exists()
    ), f"Defined checkpoint {cfg.checkpoint_path} does not exist."
    if checkpoint_path.is_dir():
        checkpoint_path = get_file_with_best_val_loss(checkpoint_path)
    return checkpoint_path


def resolve_output_dir_path(cfg: DictConfig, checkpoint_path: Path) -> Path:
    assert "out_path" in cfg, "Output path must be defined in config."
    if cfg.get("out_path", None) is not None:
        output_dir_path = (
            checkpoint_path.parents[1]
            / cfg.out_path
            / f"generated_samples_{cfg.start_time}"
        )
        output_dir_path.mkdir(parents=True, exist_ok=False)
    else:
        output_dir_path = (
            checkpoint_path.parents[1] / f"generated_samples_{cfg.start_time}"
        )
        output_dir_path.mkdir(parents=False, exist_ok=False)
    return output_dir_path


def override_hparams(
    hparams_path: Path,
    overridden_hparams: tp.Union[DictConfig, dict] = DEFAULT_OVERWRITE_HPARAMS,
) -> Path:
    if not overridden_hparams:
        return hparams_path

    if isinstance(overridden_hparams, dict):
        overridden_hparams = OmegaConf.create(overridden_hparams)

    if hparams_path.name == "hparams.original.yaml":
        # revert possibly once overwritten hparams to original before overriding
        copyfile(hparams_path, hparams_path.parent / "hparams.yaml")
        hparams_path = hparams_path.parent / "hparams.yaml"
    else:
        # backup original hparams
        copyfile(hparams_path, hparams_path.parent / "hparams.original.yaml")

    hparams = OmegaConf.load(hparams_path)
    assert type(hparams) == DictConfig, "hparams must be a DictConfig object"
    hparams = OmegaConf.merge(hparams, overridden_hparams)
    OmegaConf.save(hparams, hparams_path.parent / "hparams.yaml")
    return hparams_path.parent / "hparams.yaml"


def resolve_hparams_path(cfg: DictConfig, checkpoint_path: Path) -> Path:
    if cfg.get("hparams", None) is not None:
        hparams = Path(cfg.hparams)
        assert hparams.exists(), f"Defined hparams {cfg.hparams} does not exist."
    else:
        # try to solve hparams
        experiment_dir = checkpoint_path.parents[1]
        dirs = list(experiment_dir.iterdir())
        dirs = [
            d
            for d in dirs
            if d.is_dir()
            and not any(
                substring in d.name
                for substring in [
                    "vggsound_sparse",
                    "vggsound_test",
                    "vggsound_clean",
                    "generated_samples",
                    "visualsound",
                    "vas",
                ]
            )
        ]
        assert len(dirs) == 2  # most likely only checkpoint dir and hparams dir
        dirs.pop(dirs.index(checkpoint_path.parent))
        hparams = dirs[0] / "hparams.original.yaml"
        if not hparams.exists():
            hparams = dirs[0] / "hparams.yaml"
        assert (
            hparams.exists()
        ), f"Could not find hparams.yaml or hparams.original.yaml in {dirs[0]}"
    return hparams


def resolve_dataloader(cfg: DictConfig, duration: float) -> tp.Tuple[DataLoader, str]:
    assert cfg.get("dataloader", {}) != {}, "Dataloader config must be defined!"

    # clean up dataloader config
    datamodule_type = cfg.dataloader.pop("dataset_type")
    dataset_to_use = (cfg.dataloader.pop("dataset_to_use", "test")).lower()
    cfg.dataloader.pop("samples_per_video", None)
    cfg.dataloader.batch_size = cfg.dataloader.get("batch_size", 1)

    if "motionformer" in datamodule_type:
        cfg.dataloader["partition_video_to_clips"] = True
        cfg.dataloader["sample_duration"] = duration
    elif "vjepa" in datamodule_type:
        cfg.dataloader["partition_video_to_clips"] = False
        cfg.dataloader["sample_duration"] = duration
    else:
        cfg.dataloader["partition_video_to_clips"] = False
        cfg.dataloader["video_length"] = duration

    datamodule = get_datamodule_from_type(datamodule_type, cfg.dataloader)
    logger.info("Running datamodule setup...")
    datamodule.setup(stage=dataset_to_use)
    logger.info("Using %s dataset", dataset_to_use)
    if dataset_to_use == "train":
        dataloader = datamodule.train_dataloader()
    elif dataset_to_use == "test":
        dataloader = datamodule.test_dataloader()
    else:  # expect validation
        dataloader = datamodule.val_dataloader()
    return dataloader, datamodule_type


def generate(cfg):
    logger.setLevel(logging.INFO)
    logger.info("Generate audio with V-AURA")

    duration = cfg.get("duration", 2.56)
    generated_tokens = duration * COMPRESSION_MODEL_FRAME_RATE

    stride = cfg.get("stride", 0.64)
    assert stride % 0.64 == 0, "Stride must be a multiple of 0.64"
    vfps = cfg.get("vfps", 25)
    checkpoint_path = None
    output_dir_path = None

    device = cfg.get("device", "cuda")
    verbose = cfg.get("verbose", False)
    if verbose:
        logger.setLevel(logging.DEBUG)

    save_original_files = cfg.get("save_original_files", False)
    compress_original_audio = cfg.get("compress_original_audio", True)
    model_max_duration = cfg.get("model_max_duration", None)
    frame_step = cfg.get("frame_step", 1)

    # sampling parameters
    use_sampling = cfg.get("use_sampling", True)
    temp = cfg.get("temperature", 1.0)
    top_k = cfg.get("top_k", 256)
    top_p = cfg.get("top_p", 0.0)
    cfg_scale = cfg.get("cfg_scale", 1.0)
    audio_norm_strategy = cfg.get("audio_norm_strategy", "clip")

    # resolve necessary paths
    checkpoint_path = resolve_ckpt(cfg)
    print(checkpoint_path)
    output_dir_path = resolve_output_dir_path(cfg, checkpoint_path)
    hparams_path = override_hparams(
        resolve_hparams_path(cfg, checkpoint_path),
        cfg.get("overridden_hparams", DEFAULT_OVERWRITE_HPARAMS),
    )
    logger.info("Using checkpoint: %s", checkpoint_path.as_posix())
    logger.info("Using output dir: %s", output_dir_path.as_posix())
    logger.info("Using hparams: %s", hparams_path.as_posix())

    # load model
    with warnings.catch_warnings():  # :)
        warnings.simplefilter("ignore")
        model = VAURAModel.load_from_checkpoint(
            checkpoint_path, hparams_file=hparams_path, map_location=device
        )
    model.eval()
    # resolve model specific parameters
    if hasattr(model, "audio_encoder"):
        if model.audio_encoder.__class__.__name__ == "DacModelWrapper":
            model.sampler.audio_tokens_per_video_frame = 7
            generated_tokens = duration * COMPRESSION_MODEL_FRAME_RATE
    else:
        raise ValueError("Model must have an audio encoder.")

    if model_max_duration is None:
        if hasattr(model.sampler, "config"):  # llama
            sampler: Transformer = model.sampler
            model_max_duration = 2.56 if sampler.config.block_size > 64 else 0.64
        else:
            raise ValueError("Model max duration must be defined.")

    # resolve dataloader
    dataloader, datamodule_type = resolve_dataloader(cfg, duration)

    # resolve misc generation parameters
    video_filter = cfg.get("filter_videos", [])
    default_start_pts = 0.0
    eps = EPS_VGGSOUND if datamodule_type == "vggsound" else EPS_GREATESTHIT

    total_gen_len = int(duration * COMPRESSION_MODEL_FRAME_RATE)
    stride_tokens = int(COMPRESSION_MODEL_FRAME_RATE * stride)
    if duration >= model_max_duration:
        assert (
            stride is not None
        ), "Stride should be defined to generate beyond max_duration"
        assert (
            stride < model_max_duration
        ), "Cannot stride by more than max generation duration."

    # save generation config
    OmegaConf.save(cfg, output_dir_path / "config.yaml")

    if video_filter:
        logger.info(
            "Video filter defined: Generating %i samples and %i tokens per sample",
            len(video_filter),
            generated_tokens,
        )
    else:
        logger.info(
            "Generating %i samples and %i tokens per sample",
            len(dataloader.dataset),
            generated_tokens,
        )

    # actual generation loop
    # TODO: support prompt audio
    for sample in tqdm(dataloader):
        frames, clip_indices = None, None
        try:
            if video_filter:
                sample = {
                    key: value
                    for key, value in sample.items()
                    if value["meta"]["filepath"] in video_filter
                }
                if not sample:
                    continue

            # when using dataloader no chance to get original frames (without any transforms)
            # easily without adding extra tensor to dataloader output...
            # this is a future TODO
            original_frames, original_audios = get_original_data(
                sample["meta"],
                default_start_pts,
                eps,
                duration,
                0,  # TODO
            )
            if compress_original_audio:
                original_audios = [
                    original_audio.to(device) for original_audio in original_audios
                ]
                original_audios = [
                    original_audio.to(torch.float16)
                    for original_audio in original_audios
                ]
                original_audios = [
                    model.audio_encoder(original_audio[None])
                    for original_audio in original_audios
                ]
                original_audios = [
                    model.audio_encoder.decode([(original_audio, None)])
                    for original_audio in original_audios
                ]
            frames = sample["frames"].to(device)

            current_gen_offset: int = 0
            prompt_length: int = 0
            all_tokens = []
            prompt_tokens = None

            if duration <= model_max_duration:  # single chunk generation
                selected_frames = frames[:, :, ::frame_step, ...]
                item = model.generate(
                    frames=selected_frames,
                    audio=prompt_tokens,
                    clip_indices=clip_indices,
                    max_new_tokens=total_gen_len,
                    return_sampled_indices=True,
                    use_sampling=use_sampling,
                    temp=temp,
                    top_k=top_k,
                    top_p=top_p,
                    remove_prompts=False,
                    prompt_is_encoded=True,
                    cfg_scale=cfg_scale,
                )
                generated_audios = item["generated_audio"]

            else:  # chunked generation
                while current_gen_offset + prompt_length < total_gen_len:
                    time_offset = current_gen_offset / COMPRESSION_MODEL_FRAME_RATE
                    chunk_duration = min(duration - time_offset, model_max_duration)
                    max_gen_len = ceil(chunk_duration * COMPRESSION_MODEL_FRAME_RATE)

                    # figure out the frames to use
                    initial_position = ceil(time_offset * vfps)
                    video_target_length = ceil(chunk_duration * vfps)
                    positions = torch.arange(
                        initial_position // 16,
                        (initial_position + video_target_length) // 16,
                        device=device,
                    )
                    selected_frames = frames[:, positions % frames.shape[1], ...]
                    selected_frames = selected_frames[:, :, :, ::frame_step, ...]

                    item = model.generate(
                        frames=selected_frames,
                        audio=prompt_tokens,
                        clip_indices=clip_indices,
                        max_new_tokens=max_gen_len,
                        return_sampled_indices=True,
                        use_sampling=use_sampling,
                        temp=temp,
                        top_k=top_k,
                        top_p=top_p,
                        remove_prompts=False,
                        prompt_is_encoded=True,
                        cfg_scale=cfg_scale,
                    )
                    gen_tokens = item["sampled_indices"]
                    if prompt_tokens is None:
                        all_tokens.append(gen_tokens)
                    else:
                        all_tokens.append(gen_tokens[:, :, prompt_tokens.shape[-1] :])
                    prompt_tokens = gen_tokens[:, :, stride_tokens:]
                    prompt_length = prompt_tokens.shape[-1]
                    current_gen_offset += stride_tokens

                # gather outputs
                gen_tokens = torch.cat(all_tokens, dim=-1)
                sampled_frames = [(gen_tokens[..., : model.num_codebooks, :], None)]
                generated_audios = model.audio_encoder.decode(sampled_frames)

            for i, original_audio in enumerate(original_audios):
                save_results(
                    generated_audios[i],
                    original_frames[i],
                    output_dir_path,
                    Path(sample["meta"]["filepath"][i]).name,
                    sample["meta"]["video_fps"][i].item(),
                    44100,
                    sample["meta"]["audio_fps"][i].item(),
                    original_audio,
                    audio_norm_strategy,
                    save_original_files,
                )

        except Exception as e:
            logger.error("Error while loading sample: %s", e)
            traceback.print_exc()
            continue


def save_results(
    audio: torch.Tensor,
    frames: torch.Tensor,
    output_dir_path: Path,
    fn: str,
    v_fps: float = 25,
    generated_a_fps: int = 44100,
    original_a_fps: int = 24000,
    original_audio: tp.Optional[torch.Tensor] = None,
    audio_norm_strategy: str = "clip",
    save_original: bool = False,
):
    audio = scale_audio(audio, audio_norm_strategy, generated_a_fps)
    if fn.endswith(".mp4") or fn.endswith(".wav"):
        fn = fn[:-4]
    audio_path = output_dir_path / f"{fn}.wav"
    video_path = output_dir_path / f"{fn}.mp4"

    if video_path.exists():
        logger.warning("File %s already exists. Overwriting...", video_path.as_posix())
    write_video(
        filename=video_path.as_posix(),
        video_array=frames.permute(0, 2, 3, 1).cpu().numpy(),
        fps=v_fps,
        video_codec="h264",
        options={"crf": "10", "pix_fmt": "yuv420p"},
        audio_array=audio,
        audio_fps=generated_a_fps,
        audio_codec="aac",
    )
    torchaudio.save(str(audio_path), audio, generated_a_fps)
    logger.debug("Saved audio to %s", audio_path.as_posix())
    logger.debug("Saved video to %s", video_path.as_posix())

    if original_audio is not None and save_original:
        original_audio = scale_audio(
            original_audio, audio_norm_strategy, original_a_fps
        )
        video_path = output_dir_path / f"{fn}_original.mp4"
        write_video(
            filename=video_path.as_posix(),
            video_array=frames.permute(0, 2, 3, 1).cpu().numpy(),
            fps=v_fps,
            video_codec="h264",
            options={"crf": "10", "pix_fmt": "yuv420p"},
            audio_array=original_audio.reshape(1, -1).to("cpu"),
            audio_fps=original_a_fps,
            audio_codec="aac",
        )


def scale_audio(
    audio: torch.Tensor,
    strategy: str = "loudness",
    sample_rate: int = 24000,
    db: float = 6.0,
) -> torch.Tensor:
    if audio.dtype not in [
        torch.float32,
        torch.int32,
        torch.int16,
        torch.uint8,
    ]:
        audio = audio.to(torch.float32)
    # TODO: add different audio clipping techniques
    audio = normalize_audio(
        audio, strategy=strategy, sample_rate=sample_rate, peak_clip_headroom_db=db
    )
    audio = audio.reshape(1, -1).to("cpu")
    return audio


def process_original_data_entry(
    entry, default_start_pts, duration, eps, original_frames_list, original_audio_list
):
    start_pts = entry.get("start_pts", None)
    start_pts = start_pts.item() if start_pts else default_start_pts
    original_frames, original_audio, _ = read_video_to_frames_and_audio_streams(
        entry["filepath"],
        start_pts=start_pts,
        end_pts=start_pts + duration + eps,
    )
    original_frames = get_resize_and_convert_to_float32_transforms()(original_frames)
    original_frames = (original_frames * 255).to(torch.uint8)
    original_frames_list.append(original_frames)
    original_audio_list.append(original_audio)


def get_original_data(
    meta: dict,
    default_start_pts: float,
    eps: float,
    duration: float,
    num_processes: int = 1,
) -> tp.Tuple[list, list]:
    original_frames_list: tp.List[torch.Tensor] = []
    original_audio_list: tp.List[torch.Tensor] = []
    meta.pop("clip_indices", None)

    if num_processes > 1:
        pool = multiprocessing.Pool(processes=num_processes)
        pool.starmap(
            process_original_data_entry,
            [
                (
                    {key: value[i] for key, value in meta.items()},
                    default_start_pts,
                    duration,
                    eps,
                    original_frames_list,
                    original_audio_list,
                )
                for i in range(len(meta["filepath"]))
            ],
        )
        pool.close()
        pool.join()
    else:
        for i in range(len(meta["filepath"])):
            process_original_data_entry(
                {key: value[i] for key, value in meta.items()},
                default_start_pts,
                duration,
                eps,
                original_frames_list,
                original_audio_list,
            )

    return original_frames_list, original_audio_list


def split_into_clips(video, fpc, nc):
    """Split video into a list of clips"""
    # add one nested list for different spatial views
    # not really supported at the moment but V-JEPA expects this
    # possible e.g. add different crops from same clip (same temporally but spatially different)
    return [[video[:, i * fpc : (i + 1) * fpc]] for i in range(nc)]
