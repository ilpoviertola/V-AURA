from math import ceil

import numpy as np
import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(
        self,
        split: str,
        frame_shape: tuple = (224, 224),
        video_length: float = 2.56,
        sample_rate_audio: int = 24000,
        sample_rate_video: float = 25.0,
        frames_per_clip: int = 16,
        num_clips: int = 4,
        frame_step: int = 1,
        **kwargs,
    ) -> None:
        self.split = split
        self.frame_shape = frame_shape
        self.frames_per_clip = frames_per_clip
        self.num_clips = num_clips
        self.frame_step = frame_step

        self.video_len_in_samples = ceil(video_length * sample_rate_video)
        self.audio_len_in_samples = ceil(video_length * sample_rate_audio)

    def __len__(self) -> int:
        if self.split == "train":
            return 666
        return 66

    def __getitem__(self, idx: int) -> dict:
        video = torch.full(
            (3, self.video_len_in_samples, self.frame_shape[0], self.frame_shape[0]),
            fill_value=idx,
            dtype=torch.float32,
        )
        buffer_rgb = [
            [video[:, i * self.frames_per_clip : (i + 1) * self.frames_per_clip]]
            for i in range(self.num_clips)
        ]

        buffer_audio = torch.randn(1, self.audio_len_in_samples, dtype=torch.float32)

        meta = {
            "clip_indices": self._get_clip_indices(),
            "filepath": f"/dummy/{idx}.mp4",
        }
        return {"frames": buffer_rgb, "audio": buffer_audio, "meta": meta}

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
