import os
from pathlib import Path
from typing import Union, Optional
from datetime import datetime, timedelta
from math import sqrt
import logging

import torch
from torch import profiler as torch_profiler
from torch.nn import functional as F
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    DeviceStatsMonitor,
    BatchSizeFinder,
    LearningRateFinder,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.utilities import rank_zero_only
import numpy as np
from omegaconf import ListConfig, DictConfig

from models.data.vggsound_datamodule import VggSoundDataModule
from models.data.greatesthit_datamodule import GreatestHitDataModule
from models.data.dummy_datamodule import DummyDataModule
from models.data.video_datamodule import VideoDataModule
from models.data.vjepa_datamodule import VJEPADatamodule
from models.data.vjepa_gen_datamodule import VJEPAGenDatamodule
from models.data.audioset_datamodule import AudioSetDataModule
from models.data.motionformer_datamodule import MotionFormerDatamodule
from models.data.motionformer_gen_datamodule import MotionFormerGenDatamodule


DATALOADER_TYPES = [
    "visualsound",
    "vggsound",
    "greatesthit",
    "dummy",
    "video",
    "vjepa",
    "vjepa_gen",
    "audioset",
    "motionformer",
    "motionformer_gen",
]


def set_logging_lvl_to(lvl=logging.WARNING):
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(lvl)


def is_master(global_rank: Optional[Union[ListConfig, DictConfig, int]]):
    if type(global_rank) == int:
        return global_rank == 0
    else:  # assume OmegaConf object
        cfg = global_rank
        return cfg.trainer.global_rank == 0


def update_cfg_with_ranks(cfg: Union[ListConfig, DictConfig]):
    if "trainer" not in cfg:
        cfg.trainer = {}
    cfg.trainer.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cfg.trainer.global_rank = int(os.environ.get("RANK", 0))
    cfg.trainer.world_size = int(os.environ.get("WORLD_SIZE", 1))


def get_datamodule_from_type(dm_type: str, config: dict):
    if dm_type.lower() not in DATALOADER_TYPES:
        raise ValueError(
            f"DataModule type {dm_type} not supported. Choose from {DATALOADER_TYPES}."
        )

    config.pop("rand_transform_prob", None)

    if dm_type.lower() == "vggsound" or dm_type.lower() == "visualsound":
        return VggSoundDataModule(**config)
    elif dm_type.lower() == "greatesthit":
        return GreatestHitDataModule(**config)
    elif dm_type.lower() == "dummy":
        return DummyDataModule(**config)
    elif dm_type.lower() == "video":
        return VideoDataModule(**config)
    elif dm_type.lower() == "vjepa":
        return VJEPADatamodule(**config)
    elif dm_type.lower() == "vjepa_gen":
        return VJEPAGenDatamodule(**config)
    elif dm_type.lower() == "audioset":
        return AudioSetDataModule(**config)
    elif dm_type.lower() == "motionformer":
        return MotionFormerDatamodule(**config)
    elif dm_type.lower() == "motionformer_gen":
        return MotionFormerGenDatamodule(**config)


@rank_zero_only
def maybe_save_checkpoint(trainer: Trainer):
    print("Saving checkpoint...")
    ckpt_path = (
        Path(trainer.log_dir)
        / "checkpoints"
        / f"e{trainer.current_epoch}_last_at_{get_curr_time_w_random_shift()}.ckpt"
    )
    if trainer.global_rank == 0:
        trainer.save_checkpoint(ckpt_path)


def get_curr_time_w_random_shift() -> str:
    # shifting for a random number of seconds so that exp folder names coincide less often
    now = datetime.now() - timedelta(seconds=np.random.randint(60))
    return now.strftime("%y-%m-%dT%H-%M-%S")


# @rank_zero_only
def init_log_directory(timestamp: str, log_dir: str = "./logs") -> tuple:
    log_dir = Path(log_dir) / timestamp
    ckpt_dir = log_dir / "checkpoints"

    for d in [log_dir, ckpt_dir]:
        if not d.exists():
            d.mkdir(parents=True)
    return log_dir, ckpt_dir


def get_callbacks(
    log_dir: Path, ckpt_dir: Path, save_top_k: int = 3, early_stop_patience: int = 3
) -> list:
    if log_dir is None:  # not master process
        return []
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=ckpt_dir.as_posix(),
            filename="{epoch}-{step}-{val_loss:.3f}",
            monitor="val_loss_epoch",
            mode="min",
            save_top_k=save_top_k,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_loss_epoch",
            patience=early_stop_patience,
            mode="min",
            verbose=True,
        ),
        # DeviceStatsMonitor(cpu_stats=True),
        # BatchSizeFinder(mode='binsearch'),
        # LearningRateFinder(num_training_steps=200),
    ]
    return callbacks


def get_logger(
    log_dir: Path, name: str = "", version: str = ""
) -> Union[TensorBoardLogger, None]:
    if log_dir is None:
        return None

    logger = TensorBoardLogger(
        save_dir=log_dir.as_posix(),
        name=name,
        version=version,
        log_graph=False,
        default_hp_metric=False,
    )
    return logger


def get_profiler(
    return_profiler: str = None, log_dir: Path = None
) -> Union[None, str, PyTorchProfiler]:
    if log_dir is None or return_profiler is None:
        return None

    return_profiler = return_profiler.lower()

    if return_profiler == "pytorch":
        return PyTorchProfiler(
            on_trace_ready=torch_profiler.tensorboard_trace_handler(
                log_dir.resolve().as_posix()
            ),
            profile_memory=True,
            # TODO: Moify schedule, add skips and actives
            schedule=torch_profiler.schedule(
                skip_first=0, wait=3, warmup=3, active=15, repeat=0
            ),
        )

    if return_profiler in ["simple", "advanced", "xla"]:
        return return_profiler


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def combine_attn_weights_to_tensor(attn_weights: list) -> torch.Tensor:
    return_tensor = None
    max_new_tokens = attn_weights[-1].shape[-1]
    for w in attn_weights:
        w_padded = F.pad(w, (0, max_new_tokens - w.shape[0]), "constant", 0).unsqueeze(
            0
        )
        return_tensor = (
            w_padded
            if return_tensor is None
            else torch.cat((return_tensor, w_padded), dim=0)
        )
    return return_tensor


def generate_video_from_attn_weights(
    attn_weights: torch.Tensor, weight_size_in_px: tuple = (5, 5)
) -> torch.Tensor:
    """Generate a video from the attention weights.

    Args:
        attn_weights (Tensor): The attention weights.

    Returns:
        Tensor: The generated video.
    """
    video = torch.zeros(
        (
            1,
            attn_weights.shape[0],
            3,
            weight_size_in_px[0],
            weight_size_in_px[1] * attn_weights.shape[1],
        )
    )
    for i, w in enumerate(attn_weights):
        frame = torch.cat(
            [
                torch.full(
                    (
                        3,
                        weight_size_in_px[0],
                        weight_size_in_px[1],
                    ),
                    value,
                )
                for value in w
            ],
            dim=-1,
        )
        video[0, i, :, :, :] = frame
    return video


def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError(
            "Make sure to set the decoder_start_token_id attribute of the model's configuration."
        )
    shifted_input_ids[..., 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError(
            "Make sure to set the pad_token_id attribute of the model's configuration."
        )
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def scale_lr_with_gpu_count(lr: float, world_size: int) -> float:
    return lr * sqrt(world_size)
