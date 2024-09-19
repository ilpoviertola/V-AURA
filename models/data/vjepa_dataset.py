import typing as tp
from pathlib import Path

import torch
import numpy as np

from models.data.video_dataset import VideoDataset, VideoMeta


def get_clip_indices(
    video_len_in_samples: int,
    num_clips: int,
    frames_per_clip: int,
    frame_step: int,
    random_clip_sampling: bool = False,
    allow_clip_overlap: bool = True,
) -> list:
    # Partition video into equal sized segments and sample each clip
    # from a different segment
    partition_len = video_len_in_samples // num_clips
    clip_len = int(frames_per_clip * frame_step)

    clip_indices = []
    for i in range(num_clips):

        if partition_len > clip_len:
            # TODO: If partition_len > clip len, then sample a random window of
            # clip_len frames within the segment
            end_indx = clip_len
            if random_clip_sampling:
                end_indx = np.random.randint(clip_len, partition_len)
            start_indx = end_indx - clip_len
            indices = np.linspace(start_indx, end_indx, num=frames_per_clip)
            indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
            # --
            indices = indices + i * partition_len
        else:
            # TODO: If partition overlap not allowed and partition_len < clip_len
            # then repeatedly append the last frame in the segment until
            # we reach the desired clip length
            if allow_clip_overlap:
                indices = np.linspace(0, partition_len, num=partition_len // frame_step)
                indices = np.concatenate(
                    (
                        indices,
                        np.ones(frames_per_clip - partition_len // frame_step)
                        * partition_len,
                    )
                )
                indices = np.clip(indices, 0, partition_len - 1).astype(np.int64)
                # --
                indices = indices + i * partition_len

            # If partition overlap is allowed and partition_len < clip_len
            # then start_indx of segment i+1 will lie within segment i
            else:
                sample_len = min(clip_len, video_len_in_samples) - 1
                indices = np.linspace(0, sample_len, num=sample_len // frame_step)
                indices = np.concatenate(
                    (
                        indices,
                        np.ones(frames_per_clip - sample_len // frame_step)
                        * sample_len,
                    )
                )
                indices = np.clip(indices, 0, sample_len - 1).astype(np.int64)
                # --
                clip_step = 0
                if video_len_in_samples > clip_len:
                    clip_step = (video_len_in_samples - clip_len) // (num_clips - 1)
                indices = indices + i * clip_step

        clip_indices.append(indices)
    return clip_indices


class VJEPADataset(VideoDataset):
    def __init__(
        self,
        split: str,
        metadata: tp.List[VideoMeta],
        sample_duration: float,
        max_load_attempts: int = 10,
        filter_on_duration: bool = True,
        discarded_files: tp.List[tp.Union[str, Path]] = [],
        crop: bool = True,
        audio_transforms: tp.Optional[torch.nn.Sequential] = None,
        video_transforms: tp.Optional[torch.nn.Sequential] = None,
        partition_audio_to_clips: bool = False,
        partition_video_to_clips: bool = True,
        frames_per_clip: int = 16,
        frame_step: int = 1,
        model_fps: float = 25.0,
        assert_fps: bool = True,
    ):
        """Video dataset designed for V-JEPA feature extractor.

        Args:
            split (str): Which dataset split to use.
            metadata (tp.Dict[str, tp.Any]): Metadata for the dataset.
            sample_duration (float): Duration of single sample. The dataset can consists from videos longer than this duration.
            audio_transforms (tp.Optional[torch.nn.Sequential], optional): Audio transformations. Defaults to None.
            video_transforms (tp.Optional[torch.nn.Sequential], optional): Video transformations. Defaults to None.
            partition_audio_to_clips (bool, optional): If audio shall be partioned to V-JEPA style nested clips. Defaults to False.
            partition_video_to_clips (bool, optional): If video shall be partioned to V-JEPA style nested clips. Defaults to True.
            frames_per_clip (int, optional): How many frames V-JEPA accepts per single clip. Defaults to 16.
            frame_step (int, optional): Step between the consecutive frames. Defaults to 1.
            model_fps (float, optional): FPS V-JEPA was trained on. Defaults to 25.0.
        """
        assert frames_per_clip > 0, "Frame step must be greater than 0."
        assert frame_step > 0, "Frame step must be greater than 0."
        assert model_fps > 0, "V-JEPA FPS must be greater than 0."
        super().__init__(
            split,
            metadata,
            sample_duration,
            max_load_attempts,
            filter_on_duration,
            discarded_files,
            crop,
        )

        self.audio_transforms = audio_transforms
        self.partition_audio_to_clips = partition_audio_to_clips

        self.video_transforms = video_transforms
        self.partition_video_to_clips = partition_video_to_clips

        # V-JEPA related parameters
        self.jepa_fps = model_fps
        self.assert_fps = (
            assert_fps  # we might resample on-the-fly (generation or testing)
        )
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step

    def __getitem__(self, idx) -> dict:
        item = super().__getitem__(idx)
        if self.assert_fps:
            assert (
                item["meta"]["video_fps"] == self.jepa_fps
            ), f"Video FPS is not {self.jepa_fps}."

        item["audio"] = (
            self.audio_transforms(item["audio"])
            if self.audio_transforms is not None
            else item["audio"]
        )

        item["frames"] = (
            self.video_transforms(item["frames"])
            if self.video_transforms is not None
            else item["frames"]
        )

        if self.partition_video_to_clips:
            item = self.to_video_segments(item)

        if self.partition_audio_to_clips:
            item = self.to_audio_segments(item)

        return item

    def to_video_segments(self, item: dict) -> dict:
        """Convert a video to segments.

        Args:
            item (dict): Sample item.

        Returns:
            dict: Sample item with segmented video.
        """
        num_clips = item["frames"].shape[1] // self.frames_per_clip // self.frame_step
        assert num_clips, "num_clips is zero"
        item["frames"] = partition_video(
            item["frames"],
            self.frames_per_clip,
            self.frame_step,
            num_clips,
        )
        item["meta"]["clip_indices"] = get_clip_indices(
            video_len_in_samples=item["frames"].shape[1],
            num_clips=num_clips,
            frames_per_clip=self.frames_per_clip,
            frame_step=self.frame_step,
            random_clip_sampling=False,  # TODO: implement this
            allow_clip_overlap=True,
        )
        return item

    def to_audio_segments(self, item: dict) -> dict:
        """Convert audio to segments.

        Args:
            item (dict): Sample item.

        Returns:
            dict: Sample item with segmented audio.
        """
        num_clips = item["frames"].shape[1] // self.frames_per_clip // self.frame_step
        assert num_clips, "num_clips is zero"
        item["audio"] = partition_audio(
            item["audio"],
            self.frames_per_clip,
            self.frame_step,
            item["meta"]["video_fps"],
            item["meta"]["audio_fps"],
            num_clips,
        )
        return item


def partition_video(
    video: torch.Tensor,
    frames_per_clip: int,
    frame_step: int,
    num_clips: int,
):
    return [
        [
            video[
                :,
                i
                * (frames_per_clip * frame_step) : (i + 1)
                * (frames_per_clip * frame_step) : frame_step,
            ]
        ]
        for i in range(num_clips)
    ]


def partition_audio(
    audio: torch.Tensor,
    frames_per_clip: int,
    frame_step: int,
    vfps: float,
    afps: float,
    num_clips: int,
):
    ratio = frames_per_clip / vfps
    fpc = int(ratio * afps * frame_step)
    return [[audio[:, i * fpc : (i + 1) * fpc]] for i in range(num_clips)]
