import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np

from models.data.dummy_dataset import DummyDataset

class DummyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        frame_shape: tuple = (224, 224),
        **kwargs,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.frame_shape = frame_shape
        self.datasets = {}

    def setup(self, stage=None):
        for split in ["train", "validation", "test", "predict"]:
            self.datasets[split] = DummyDataset(
                split,
                frame_shape=self.frame_shape,
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
