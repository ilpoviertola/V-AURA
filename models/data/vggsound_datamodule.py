from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from models.data.vggsound_dataset import VggSoundDataset


class VggSoundDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        split_dir: str,
        meta_file: str,
        batch_size: int,
        num_workers: int,
        run_additional_checks: bool = True,
        excluded_files: str = None,
        included_files: str = None,
        fixed_start_pts_file: str = None,
        pin_memory: bool = False,
        computing_mean_std: bool = False,
        partition_audio_to_clips: bool = False,
        audio_transforms_train: list = [],
        audio_transforms_test: list = [],
        video_transforms_train: list = [],
        video_transforms_test: list = [],
        **kwargs,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split_dir = Path(split_dir)
        self.meta_file = Path(meta_file)
        self.excluded_files = Path(excluded_files) if excluded_files else None
        self.included_files = Path(included_files) if included_files else None
        self.fixed_start_pts_file = (
            Path(fixed_start_pts_file) if fixed_start_pts_file else None
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.run_additional_checks = run_additional_checks
        self.pin_memory = pin_memory
        self.computing_mean_std = computing_mean_std
        self.partition_audio_to_clips = partition_audio_to_clips
        self.audio_transforms_train = audio_transforms_train
        self.audio_transforms_test = audio_transforms_test
        self.video_transforms_train = video_transforms_train
        self.video_transforms_test = video_transforms_test

        self.kwargs = kwargs

        assert (
            self.data_dir.exists() and self.data_dir.is_dir()
        ), f"Data directory {self.data_dir.as_posix()} does not exist."
        assert (
            self.split_dir.exists() and self.split_dir.is_dir()
        ), f"Split directory {self.split_dir.as_posix()} does not exist."
        assert (
            self.meta_file.exists() and self.meta_file.is_file()
        ), f"Meta file {self.meta_file.as_posix()} does not exist."
        if self.excluded_files is not None:
            assert (
                self.excluded_files.exists()
            ), f"Excluded files path (file/dir) {self.excluded_files.as_posix()} does not exist."
        if self.included_files is not None:
            assert (
                self.included_files.exists()
            ), f"Included files path (file/dir) {self.included_files.as_posix()} does not exist."
        assert batch_size > 0, f"Batch size ({self.batch_size}) must be greater than 0."
        assert (
            num_workers >= 0
        ), f"Number of workers ({self.num_workers}) must be equal or greater than 0."

        self.datasets = {}

    def setup(self, stage=None):
        for split in ["train", "validation", "test", "predict"]:
            self.datasets[split] = VggSoundDataset(
                split=split,
                split_dir_path=self.split_dir,
                data_path=self.data_dir,
                meta_path=self.meta_file,
                run_additional_checks=self.run_additional_checks,
                excluded_files_path=self.excluded_files,
                included_files_path=self.included_files,
                fixed_start_pts_file_path=self.fixed_start_pts_file,
                computing_mean_std=self.computing_mean_std,
                audio_transforms=(
                    self.audio_transforms_train
                    if split == "train"
                    else self.audio_transforms_test
                ),
                video_transforms=(
                    self.video_transforms_train
                    if split == "train"
                    else self.video_transforms_test
                ),
                partition_audio_to_clips=self.partition_audio_to_clips,
                **self.kwargs,
            )

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            worker_init_fn=self.worker_init_fn,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=self.worker_init_fn,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=self.worker_init_fn,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.datasets["predict"],
            batch_size=1,
            num_workers=self.num_workers,
            worker_init_fn=self.worker_init_fn,
            pin_memory=self.pin_memory,
        )

    @staticmethod
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
