import typing as tp
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from torch.nn import Sequential
from torch.utils.data import DataLoader

from models.data.video_dataset import VideoDataset


class VideoDataModule(pl.LightningDataModule):
    DATASET = VideoDataset

    def __init__(
        self,
        path_to_metadata: tp.Union[Path, str],
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        assert Path(
            path_to_metadata
        ).exists(), f"Path {path_to_metadata} does not exist."
        assert Path(
            path_to_metadata
        ).is_dir(), f"Path {path_to_metadata} is not a directory."
        assert batch_size > 0, f"Batch size must be greater than 0."
        assert (
            num_workers >= 0
        ), f"Number of workers must be greater than or equal to 0."

        self.path_to_metadata = Path(path_to_metadata)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.args = args
        self.kwargs = kwargs
        self.datasets: tp.Dict[str, self.DATASET] = {}

    def setup(self, stage: tp.Optional[str] = None):
        """Setup the data module.

        Args:
            stage (tp.Optional[str], optional): 'fit', 'validate', 'test', or 'predict'. Defaults to None.
        """
        if stage == "fit":
            self.setup_train()
            self.setup_validation()
            self.setup_predict()  # we predict at validation start

        if stage == "validate":
            self.setup_predict()

        if stage == "test":
            self.setup_test()

        if stage == "predict":
            self.setup_predict()

    def setup_train(self):
        if "train" not in self.datasets:
            self.kwargs["split"] = "train"
            self.datasets["train"] = self.DATASET.from_meta_file(
                self.path_to_metadata / "train", *self.args, **self.kwargs
            )

    def setup_validation(self):
        if "validation" not in self.datasets:
            self.kwargs["split"] = "validation"
            self.datasets["validation"] = self.DATASET.from_meta_file(
                self.path_to_metadata / "validation", *self.args, **self.kwargs
            )

    def setup_predict(self):
        if "predict" not in self.datasets:
            self.kwargs["split"] = "predict"
            self.datasets["predict"] = self.DATASET.from_meta_file(
                self.path_to_metadata / "predict", *self.args, **self.kwargs
            )

    def setup_test(self):
        if "test" not in self.datasets:
            self.kwargs["split"] = "test"
            self.datasets["test"] = self.DATASET.from_meta_file(
                self.path_to_metadata / "test", *self.args, **self.kwargs
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
