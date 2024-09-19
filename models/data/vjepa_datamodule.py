import typing as tp
from pathlib import Path

from torch.nn import Sequential

from models.data.video_datamodule import VideoDataModule
from models.data.vjepa_dataset import VJEPADataset
from models.data.transforms.video_transforms import get_video_transforms
from models.data.transforms.audio_transforms import get_audio_transforms


class VJEPADatamodule(VideoDataModule):
    DATASET = VJEPADataset

    def __init__(
        self,
        path_to_metadata: tp.Union[Path, str],
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
        audio_transforms_train: list = [],
        audio_transforms_test: list = [],
        video_transforms_train: list = [],
        video_transforms_test: list = [],
        *args,
        **kwargs,
    ):
        super().__init__(
            path_to_metadata,
            batch_size,
            num_workers,
            pin_memory,
            *args,
            **kwargs,
        )

        self.audio_transforms_train = audio_transforms_train
        self.inited_audio_transforms_train: tp.Optional[Sequential] = None
        self.audio_transforms_test = audio_transforms_test
        self.inited_audio_transforms_test: tp.Optional[Sequential] = None
        self.video_transforms_train = video_transforms_train
        self.inited_video_transforms_train: tp.Optional[Sequential] = None
        self.video_transforms_test = video_transforms_test
        self.inited_video_transforms_test: tp.Optional[Sequential] = None

    def setup(self, stage: tp.Optional[str] = None):
        """Setup the data module.

        Args:
            stage (tp.Optional[str], optional): 'fit', 'validate', 'test', or 'predict'. Defaults to None.
        """
        if self.inited_audio_transforms_train is None or len(self.inited_audio_transforms_train) == 0:
            self.inited_audio_transforms_train = get_audio_transforms(
                self.audio_transforms_train
            )
        if self.inited_audio_transforms_test is None or len(self.inited_audio_transforms_test) == 0:
            self.inited_audio_transforms_test = get_audio_transforms(
                self.audio_transforms_test
            )
        if self.inited_video_transforms_train is None or len(self.inited_video_transforms_train) == 0:
            self.inited_video_transforms_train = get_video_transforms(
                self.video_transforms_train
            )
        if self.inited_video_transforms_test is None or len(self.inited_video_transforms_test) == 0:
            self.inited_video_transforms_test = get_video_transforms(
                self.video_transforms_test
            )
        super().setup(stage)

    def setup_train(self):
        if "train" not in self.datasets:
            self.kwargs["split"] = "train"
            self.kwargs["audio_transforms"] = self.inited_audio_transforms_train
            self.kwargs["video_transforms"] = self.inited_video_transforms_train
            self.datasets["train"] = self.DATASET.from_meta_file(
                self.path_to_metadata / "train", *self.args, **self.kwargs
            )

    def setup_validation(self):
        if "validation" not in self.datasets:
            self.kwargs["split"] = "validation"
            self.kwargs["audio_transforms"] = self.inited_audio_transforms_test
            self.kwargs["video_transforms"] = self.inited_video_transforms_test
            self.datasets["validation"] = self.DATASET.from_meta_file(
                self.path_to_metadata / "validation", *self.args, **self.kwargs
            )

    def setup_predict(self):
        if "predict" not in self.datasets:
            self.kwargs["split"] = "predict"
            self.kwargs["audio_transforms"] = self.inited_audio_transforms_test
            self.kwargs["video_transforms"] = self.inited_video_transforms_test
            self.datasets["predict"] = self.DATASET.from_meta_file(
                self.path_to_metadata / "predict", *self.args, **self.kwargs
            )

    def setup_test(self):
        if "test" not in self.datasets:
            self.kwargs["split"] = "test"
            self.kwargs["audio_transforms"] = self.inited_audio_transforms_test
            self.kwargs["video_transforms"] = self.inited_video_transforms_test
            self.datasets["test"] = self.DATASET.from_meta_file(
                self.path_to_metadata / "test", *self.args, **self.kwargs
            )
