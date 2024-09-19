from pathlib import Path
from typing import Optional, Dict, List, Union
import csv
from collections import Counter
from math import ceil, floor
import json

# import torch
import numpy as np
from torch.utils.data import Dataset

from utils.data_utils import (
    read_video_to_frames_and_audio_streams,
    # read_video_to_frames_and_audio_streams_with_decord,
)
from models.data.transforms.video_transforms import (
    get_s3d_transforms,
    get_video_transforms,
    GenerateMultipleSegments,
)
from models.data.transforms.audio_transforms import get_audio_transforms


EPS = (
    0.01  # torch.finfo(torch.float32).eps # this is pretty large but works the best...
)


class VggSoundDataset(Dataset):
    def __init__(
        self,
        split: str,
        split_dir_path: Path,
        data_path: Path,
        meta_path: Path,
        excluded_files_path: Optional[Path] = None,
        included_files_path: Optional[Path] = None,
        fixed_start_pts_file_path: Optional[Path] = None,
        video_length: float = 2.56,
        sample_rate_audio: int = 24000,
        sample_rate_video: float = 25.0,
        audio_transforms: list = [],
        video_transforms: list = [],
        run_additional_checks: bool = True,
        computing_mean_std: bool = False,
        original_video_file_len: float = 10.0,  # this is the length of original MP4 file
        frames_per_clip: int = 16,
        num_clips: int = 4,
        frame_step: int = 1,
        partition_audio_to_clips: bool = False,
        partition_video_to_clips: bool = True,
        filter_by_imagebind_score: bool = False,
        imagebind_score_threshold: float = 0.0,
        imagebind_score_file_path: Optional[str] = None,
        filter_by_insync: bool = False,
        insync_filter_key: str = "is_correct",  # is_correct or is_correct_within_1cls_tol
        insync_filter_threshold: int = -1,
        insync_file_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.split = split

        split_prefix = (
            "vggsound" if "vggsound" in split_dir_path.name else "visualsound"
        )
        self.split_file_path = split_dir_path / f"{split_prefix}_{self.split}.txt"
        assert (
            self.split_file_path.is_file()
        ), f"Split file {self.split_file_path.as_posix()} does not exist."

        self.data_path = data_path
        assert (
            self.data_path.is_dir()
        ), f"Data directory {self.data_path.as_posix()} does not exist."

        self.meta_path = meta_path
        assert (
            self.meta_path.is_file()
        ), f"Meta file {self.meta_path.as_posix()} does not exist."

        if excluded_files_path is not None:
            assert (
                excluded_files_path.exists()
            ), f"Excluded files (file/dir) {excluded_files_path.as_posix()} does not exist."

        if included_files_path is not None:
            assert (
                included_files_path.exists()
            ), f"Included files (file/dir) {included_files_path.as_posix()} does not exist."

        if fixed_start_pts_file_path is None:
            fixed_start_pts = {}
        else:
            assert (
                fixed_start_pts_file_path.exists()
            ), f"Fixed start points file {fixed_start_pts_file_path.as_posix()} does not exist."
            with open(fixed_start_pts_file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=",")
                next(reader)  # skip header
                fixed_start_pts = {row[0]: float(row[1]) for row in reader}
        self.fixed_start_pts = fixed_start_pts

        self.a_sr = sample_rate_audio
        self.v_sr = sample_rate_video
        self.video_len = video_length
        self.video_len_in_samples = ceil(video_length * sample_rate_video)
        self.audio_len_in_samples = ceil(video_length * sample_rate_audio)
        self.original_video_file_len = original_video_file_len
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = floor(
            (self.video_len_in_samples / frame_step) / frames_per_clip
        )

        meta = list(csv.reader(self.meta_path.open("r")))[1:]  # skip header
        unique_classes = sorted(list(set(row[2] for row in meta)))
        self.label2target = {
            label: target for target, label in enumerate(unique_classes)
        }
        self.target2label = {
            target: label for label, target in self.label2target.items()
        }
        self.video2target = {row[0]: self.label2target[row[2]] for row in meta}

        meta_available = set(
            [f"{r[0]}_{int(r[1])*1000}_{(int(r[1])+10)*1000}" for r in meta]
        )
        with open(self.split_file_path, encoding="utf-8") as f:
            within_split = set(f.read().splitlines())

        meta_available = meta_available.intersection(within_split)

        if excluded_files_path is not None:
            excluded = self._get_bad_examples(excluded_files_path)
            meta_available = meta_available.difference(excluded)

        if included_files_path is not None:
            included = self._get_good_examples(included_files_path)
            meta_available = meta_available.intersection(included)

        # do not change test/val splits so the results are comparable
        if filter_by_imagebind_score and self.split != "predict":
            assert (
                imagebind_score_file_path is not None
            ), "Imagebind score file path is required for filtering by imagebind score."
            ib_score_file_path = Path(imagebind_score_file_path)
            assert (
                ib_score_file_path.is_file()
            ), f"Imagebind score file {ib_score_file_path.as_posix()} does not exist."
            excluded = self._get_bad_imagebind_examples(
                ib_score_file_path, imagebind_score_threshold
            )
            meta_available = meta_available.difference(excluded)

        # do not change test/val splits so the results are comparable
        if filter_by_insync and self.split != "predict":
            assert insync_file_path is not None, "Insync file path is required."
            assert Path(insync_file_path).is_file(), f"Insync file does not exist."
            excluded = self._get_bad_insync_examples(
                Path(insync_file_path), insync_filter_key, insync_filter_threshold
            )
            meta_available = meta_available.difference(excluded)

        clip_paths = [
            self.data_path / f"{v}.mp4"
            for v in meta_available.intersection(within_split)
        ]

        # clip_paths = sorted(clip_paths)  # TODO: Think about sorting
        self.partition_audio_to_clips = partition_audio_to_clips
        self.partition_video_to_clips = partition_video_to_clips
        self.dataset = clip_paths
        if run_additional_checks:
            # Making sure that all classes have at least one example
            counter = Counter(
                [self.video2target[Path(p).stem[:11]] for p in clip_paths]
            )
            assert all(
                counter[c] > 0 for c in self.target2label.keys()
            ), f"Some classes have 0 count: {dict(counter)}"
            # Test that provided video and audio sample rates match to actual data
            self._test_video_sample_rate_match()

        # Transforms
        self.audio_transforms = (
            get_audio_transforms(audio_transforms) if audio_transforms else None
        )
        self.video_transforms = (
            get_video_transforms(video_transforms)
            if video_transforms
            else get_s3d_transforms()
        )
        self.to_segments_transform = GenerateMultipleSegments(
            segment_size_vframes=self.frames_per_clip,
            n_segments=self.num_clips,
            is_start_random=(self.split == "train"),
            audio_jitter_sec=0.0,
            step_size_seg=self.frame_step,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        if self.split == "train":
            # if we are training, randomly sample a starting point from the self.original_video_file_len sec video
            start_pts = np.random.uniform(
                0, self.original_video_file_len - self.video_len - EPS
            )
        else:
            if self.video_len > 5.12:
                start_pts = 0.0
            else:
                start_pts = self.fixed_start_pts.get(self.dataset[idx].stem, 0.0)

        # sometimes loading these videos is pain in the ass due to the missing v/a frames...
        loaded_video = False
        data_path: str = self.dataset[idx].resolve().as_posix()
        while not loaded_video:
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

        rgb = rgb[: self.video_len_in_samples : self.frame_step, ...]
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

        # generate segments for MotionFormer
        new_meta = {
            "video": {"fps": [meta["video_fps"]]},
            "audio": {"framerate": [meta["audio_fps"]]},
        }
        meta["filepath"] = data_path
        meta["clip_indices"] = self._get_clip_indices()
        meta["start_pts"] = start_pts
        item = {
            "video": buffer_rgb,
            "audio": buffer_audio,
            "path": data_path,
            "meta": new_meta,
        }
        item = self.to_segments_transform(item)
        return {
            "frames": item["video"].permute(0, 2, 1, 3, 4),
            "audio": item["audio"].reshape(1, -1),
            "meta": meta,
        }

    def _test_video_sample_rate_match(self):
        data_path: str = self.dataset[0].as_posix()
        _, _, meta = read_video_to_frames_and_audio_streams(
            data_path, end_pts=self.video_len
        )
        assert (
            meta["video_fps"] == self.v_sr
        ), f"Video sample rate mismatch. User provided video FPS: {self.v_sr}, but actual video FPS: {meta['video_fps']}"
        assert (
            meta["audio_fps"] == self.a_sr
        ), f"Audio sample rate mismatch. User provided audio SR: {self.a_sr}, but actual audio SR: {meta['audio_fps']}"

    def _get_bad_examples(self, exluded_files_path: Path) -> set:
        if exluded_files_path.is_file():
            with open(exluded_files_path, encoding="utf-8") as f:
                bad = set(f.read().splitlines())
            return bad
        else:  # expect dir
            bad = set()
            for p in exluded_files_path.glob("*.txt"):
                with open(p, encoding="utf-8") as f:
                    bad = bad.union(set(f.read().splitlines()))
            return bad

    def _get_good_examples(self, included_files_path: Path) -> set:
        if included_files_path.is_file():
            with open(included_files_path, encoding="utf-8") as f:
                if included_files_path.suffix == ".csv":
                    reader = csv.reader(f)
                    next(reader)  # skip header
                    good = set(row[0] for row in reader)
                else:
                    good = set(f.read().splitlines())
            return good
        else:
            good = set()
            for p in included_files_path.glob("*.txt"):
                with open(p, encoding="utf-8") as f:
                    good = good.union(set(f.read().splitlines()))
            return good

    def _get_bad_imagebind_examples(
        self, imagebind_score_file_path: Path, threshold: float
    ) -> set:
        with open(imagebind_score_file_path, "r", encoding="utf-8") as f:
            ib_scores = json.load(f)

        return set(Path(k).stem for k, v in ib_scores.items() if v < threshold)

    def _get_bad_insync_examples(
        self, insync_file_path: Path, insync_key: str, threshold: int
    ) -> set:
        insync_key = insync_key.lower()
        assert insync_key in [
            "is_correct",
            "is_correct_within_1cls_tol",
        ], "Invalid insync key."
        if threshold < 0:
            threshold = 25 if self.split == "train" else 5  # default thresholds
        insync_data = {}
        with open(insync_file_path, "r", encoding="utf-8") as f:
            csvreader = csv.reader(f)
            for row in csvreader:
                if row[0] not in insync_data:
                    insync_data[row[0]] = {
                        "offset": [row[1]],
                        "vstart": [row[2]],
                        "is_correct": [int(row[3])],
                        "is_correct_within_1cls_tol": [int(row[4])],
                        "split": Path(insync_file_path).stem.split("_")[-1],
                    }
                else:
                    insync_data[row[0]]["offset"].append(row[1])  # type: ignore
                    insync_data[row[0]]["vstart"].append(row[2])  # type: ignore
                    insync_data[row[0]]["is_correct"].append(int(row[3]))  # type: ignore
                    insync_data[row[0]]["is_correct_within_1cls_tol"].append(int(row[4]))  # type: ignore
        return set(
            [
                video
                for video, meta in insync_data.items()
                if sum(meta[insync_key]) < threshold  # type: ignore
            ]
        )

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
