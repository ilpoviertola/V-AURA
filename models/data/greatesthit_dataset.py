"""
This dataset expects the following directory structure:
    - data_path
        - audio
            - *.pt  # raw audio as tensor (preprocess_greatest_hit.py)
        - video
            - *.pt  # raw video as tensor (preprocess_greatest_hit.py)
    - meta_path
        - metadata.csv
    - split_dir_path
        - greatesthit_{split}.txt
"""

from pathlib import Path
import csv
from math import ceil
from typing import Dict

import numpy as np
from einops import repeat
from torch.nn import functional as F
from torch.utils.data import Dataset

from models.data.transforms.video_transforms import (
    get_s3d_transforms,
    get_s3d_transforms_validation,
    get_video_transforms,
)
from models.data.transforms.audio_transforms import get_audio_transforms
from utils.data_utils import read_video_to_frames_and_audio_streams, loadvideo_decord


EPS = 1e-9


class GreatestHitDataset(Dataset):
    def __init__(
        self,
        split: str,
        split_dir_path: Path,
        data_path: Path,
        meta_path: Path,
        video_length: float = 2.56,
        sample_rate_audio: int = 24000,
        sample_rate_video: float = 25.0,
        audio_transforms: list = [],
        video_transforms: list = [],
        run_additional_checks: bool = True,
        sample_single_frame: bool = False,
        frames_per_clip: int = 16,
        num_clips: int = 4,
        frame_step: int = 1,
        partition_audio_to_clips: bool = False,
        computing_mean_std: bool = False,
        excluded_files_path: Path = None,
        partition_video_to_clips: bool = True,
        original_video_file_len: float = 5.00,
        **kwargs,
    ) -> None:
        super().__init__()
        self.split = split

        split_file_path = split_dir_path / f"greatesthit_{self.split}.txt"
        assert (
            split_file_path.is_file()
        ), f"Split file {split_file_path.as_posix()} does not exist."

        self.data_path = data_path
        assert (
            data_path.is_dir()
        ), f"Data directory {data_path.as_posix()} does not exist."

        assert meta_path.is_file(), f"Meta file {meta_path.as_posix()} does not exist."

        self.dataset = []
        with open(split_file_path, encoding="utf-8") as f:
            within_split = f.read().splitlines()

        for basename in within_split:
            files = self._get_all_files_with_same_basename(basename, data_path)
            self.dataset += files

        self.dataset = [data_path / f for f in self.dataset]

        (
            self.filename2label,
            self.filename2material,
            self.filename2motion,
        ) = self._get_filename2all(meta_path)

        self.video_len_in_samples = ceil(video_length * sample_rate_video)
        self.audio_len_in_samples = ceil(video_length * sample_rate_audio)
        self.video_len = video_length
        self.a_sr = sample_rate_audio
        self.v_sr = sample_rate_video
        self.sample_single_frame = sample_single_frame
        if self.sample_single_frame:
            self.frame_num = (
                self.video_len_in_samples // 2
                if self.video_len_in_samples % 2 == 0
                else (self.video_len_in_samples - 1) // 2
            )
        self.frames_per_clip = frames_per_clip
        self.num_clips = num_clips
        self.frame_step = frame_step

        # TODO: Fix this hack
        self.audio_transforms = (
            get_audio_transforms(audio_transforms) if audio_transforms else None
        )
        self.video_transforms = (
            get_video_transforms(video_transforms)
            if video_transforms
            else get_s3d_transforms()
        )
        self.partition_audio_to_clips = partition_audio_to_clips
        self.partition_video_to_clips = partition_video_to_clips
        self.original_video_file_len = original_video_file_len
        self.fixed_start_pts: Dict[str, float] = {}

        if run_additional_checks:
            pass  # for now

    def __getitem__(self, idx) -> dict:
        if self.split == "train":
            # if we are training, randomly sample a starting point from the self.original_video_file_len sec video
            start_pts = np.random.uniform(
                0, self.original_video_file_len - self.video_len - EPS
            )

        # sometimes loading these videos is pain in the ass due to the missing v/a frames...
        loaded_video = False
        data_path: str = self.dataset[idx].resolve().as_posix()
        while not loaded_video:
            if self.split != "train":
                if self.dataset[idx].stem not in self.fixed_start_pts:
                    self.fixed_start_pts[self.dataset[idx].stem] = np.random.uniform(
                        0, self.original_video_file_len - self.video_len - EPS
                    )
                start_pts = self.fixed_start_pts[self.dataset[idx].stem]
            # try to read a EPS seconds more video and then crop to desired sample lens
            rgb, audio, meta = read_video_to_frames_and_audio_streams(
                data_path, start_pts=start_pts, end_pts=start_pts + self.video_len + EPS
            )
            loaded_video = (
                rgb.shape[0] >= self.video_len_in_samples
                and audio.shape[-1] >= self.audio_len_in_samples
            )
            if not loaded_video:
                idx = np.random.randint(self.__len__())
                data_path = self.dataset[idx].resolve().as_posix()

        rgb = rgb[: self.video_len_in_samples, ...]
        rgb = self.video_transforms(rgb) if self.video_transforms else rgb

        audio = audio[..., : self.audio_len_in_samples]
        audio = self.audio_transforms(audio) if self.audio_transforms else audio

        def split_into_clips(video):
            """Split video into a list of clips"""
            fpc = self.frames_per_clip
            nc = self.num_clips
            # add one nested list for different spatial views
            # not really supported at the moment but V-JEPA expects this
            # possible e.g. add different crops from same clip (same temporally but spatially different)
            return [[video[:, i * fpc : (i + 1) * fpc]] for i in range(nc)]

        def split_into_audio_clips(audio):
            """Split video into a list of clips"""
            ratio = self.frames_per_clip / self.v_sr
            fpc = int(ratio * self.a_sr)
            nc = self.num_clips
            return [[audio[:, i * fpc : (i + 1) * fpc]] for i in range(nc)]

        buffer_rgb = split_into_clips(rgb) if self.partition_video_to_clips else rgb
        buffer_audio = (
            split_into_audio_clips(audio) if self.partition_audio_to_clips else audio
        )

        meta["filepath"] = data_path
        meta["clip_indices"] = self._get_clip_indices()
        meta["start_pts"] = start_pts
        meta["label"] = self.filename2label[self.dataset[idx].name]
        meta["material"] = self.filename2material[self.dataset[idx].name]
        meta["motion"] = self.filename2motion[self.dataset[idx].name]
        return {"frames": buffer_rgb, "audio": buffer_audio, "meta": meta}

    def __len__(self) -> int:
        return len(self.dataset)

    def _get_all_files_with_same_basename(self, basename: str, data_dir: Path) -> list:
        all_files = (
            data_dir.glob(f"{basename}_denoised*")
            if self.split != "predict"
            else data_dir.glob(f"{basename}*")
        )
        return [f.name for f in list(all_files)]  # return only filenames

    def _get_clip_indices(
        self, random_clip_sampling: bool = False, allow_clip_overlap: bool = True
    ) -> list:
        # Partition video into equal sized segments and sample each clip
        # from a different segment
        partition_len = self.video_len_in_samples // self.num_clips
        clip_len = int(self.frames_per_clip * self.frame_step)

        clip_indices = []
        for i in range(self.num_clips):

            if partition_len > clip_len:
                # TODO: If partition_len > clip len, then sample a random window of
                # clip_len frames within the segment
                end_indx = clip_len
                if random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=self.frames_per_clip)
                indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
                # --
                indices = indices + i * partition_len
            else:
                # TODO: If partition overlap not allowed and partition_len < clip_len
                # then repeatedly append the last frame in the segment until
                # we reach the desired clip length
                if allow_clip_overlap:
                    indices = np.linspace(
                        0, partition_len, num=partition_len // self.frame_step
                    )
                    indices = np.concatenate(
                        (
                            indices,
                            np.ones(
                                self.frames_per_clip - partition_len // self.frame_step
                            )
                            * partition_len,
                        )
                    )
                    indices = np.clip(indices, 0, partition_len - 1).astype(np.int64)
                    # --
                    indices = indices + i * partition_len

                # If partition overlap is allowed and partition_len < clip_len
                # then start_indx of segment i+1 will lie within segment i
                else:
                    sample_len = min(clip_len, self.video_len_in_samples) - 1
                    indices = np.linspace(
                        0, sample_len, num=sample_len // self.frame_step
                    )
                    indices = np.concatenate(
                        (
                            indices,
                            np.ones(
                                self.frames_per_clip - sample_len // self.frame_step
                            )
                            * sample_len,
                        )
                    )
                    indices = np.clip(indices, 0, sample_len - 1).astype(np.int64)
                    # --
                    clip_step = 0
                    if self.video_len_in_samples > clip_len:
                        clip_step = (self.video_len_in_samples - clip_len) // (
                            self.num_clips - 1
                        )
                    indices = indices + i * clip_step

            clip_indices.append(indices)
        return clip_indices

    @staticmethod
    def _get_filename2all(meta_path: Path) -> tuple:
        filename2label = {}
        filename2material = {}
        filename2motion = {}
        with open(meta_path, encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # skip header
            for row in reader:
                filename2label[row[0]] = row[5]
                filename2material[row[0]] = row[4]
                filename2motion[row[0]] = row[6]
        return filename2label, filename2material, filename2motion

    @staticmethod
    def _get_max_len_in_samples(filepath: Path) -> int:
        with open(filepath, encoding="utf-8") as f:
            return int(f.read().strip())

    @staticmethod
    def _get_base_file_name(filename: str) -> str:
        return "_".join(filename.split("_")[:-1])
