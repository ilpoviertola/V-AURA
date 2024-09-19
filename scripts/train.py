"""Training script for the model. Use through main.py."""

import logging
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
import traceback

from models.vaura_model import VAURAModel
from utils.train_utils import (
    get_callbacks,
    init_log_directory,
    get_logger,
    maybe_save_checkpoint,
    get_profiler,
    set_logging_lvl_to,
    is_master,
)
from utils.train_utils import get_datamodule_from_type, scale_lr_with_gpu_count


def train(cfg: OmegaConf):
    # Initialize log directories
    if cfg.trainer.ckpt_path and is_master(cfg):
        log_dir = Path(cfg.trainer.ckpt_path).parents[1]
        ckpt_dir = Path(cfg.trainer.ckpt_path).parents[0]
    else:
        log_dir, ckpt_dir = init_log_directory(cfg.start_time, cfg.trainer.log_dir)
    # TODO: Implement config validation
    datamodule_type = cfg.dataloader.pop("dataset_type")
    datamodule = get_datamodule_from_type(datamodule_type, cfg.dataloader)
    # scale learning rate
    cfg.model.learning_rate = (
        scale_lr_with_gpu_count(cfg.model.learning_rate, cfg.trainer.world_size)
        if cfg.trainer.scale_lr_with_gpu_count and cfg.trainer.world_size > 1
        else cfg.model.learning_rate
    )

    model = VAURAModel(
        learning_rate=cfg.model.learning_rate,
        lr_scheduler=(
            OmegaConf.to_container(cfg.model.lr_scheduler)
            if cfg.model.lr_scheduler is not None
            else None
        ),
        weight_decay=cfg.model.weight_decay,
        betas=cfg.model.betas,
        batch_size=cfg.model.batch_size,
        use_visual_conditioning=cfg.model.use_visual_conditioning,
        feature_extractor_config=cfg.model.feature_extractor_config,
        audio_encoder_config=cfg.model.audio_encoder_config,
        sampler_config=cfg.model.sampler_config,
        visual_bridge_config=cfg.model.visual_bridge_config,
        pattern_provider_config=cfg.model.pattern_provider_config,
        predict_at_val_start=cfg.model.predict_at_val_start,
        return_attention_weights=cfg.model.return_attention_weights,
        plot_distr_of_pred_indices=cfg.model.plot_distr_of_pred_indices,
        freeze_feature_extractor=cfg.model.freeze_feature_extractor,
        files_to_track_during_training=cfg.model.files_to_track_during_training,
        flatten_vis_feats=cfg.model.flatten_vis_feats,
        apply_per_video_frame_mask=cfg.model.apply_per_video_frame_mask,
        lora_finetune_feature_extractor=cfg.model.lora_finetune_feature_extractor,
        lora_target_modules=cfg.model.lora_target_modules,
        lora_rank=cfg.model.lora_rank,
    )

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        strategy=cfg.trainer.strategy,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        min_epochs=cfg.trainer.min_epochs,
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        overfit_batches=cfg.trainer.overfit_batches,
        val_check_interval=cfg.trainer.val_check_interval,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        gradient_clip_algorithm=cfg.trainer.gradient_clip_algorithm,
        fast_dev_run=cfg.trainer.fast_dev_run,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        limit_test_batches=cfg.trainer.limit_test_batches,
        log_every_n_steps=(1 if cfg.trainer.overfit_batches else 25),
        profiler=get_profiler(cfg.trainer.profiler, log_dir),
        callbacks=get_callbacks(
            log_dir, ckpt_dir, early_stop_patience=cfg.trainer.early_stop_patience
        ),
        logger=get_logger(log_dir, name=cfg.trainer.experiment_name),
        benchmark=cfg.trainer.benchmark,
    )

    try:
        set_logging_lvl_to(logging.INFO)
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.trainer.ckpt_path)
        trainer.test(model=model, datamodule=datamodule, ckpt_path="best")
    except BaseException as e:
        print(e)
        traceback.print_exc()
        maybe_save_checkpoint(trainer)
