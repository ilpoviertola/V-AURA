import math
import random

import torch
import torchvision

torchvision.disable_beta_transforms_warning()
import torchvision.transforms as Tv
from torchvision.transforms import v2

from utils.utils import instantiate_from_config


def sec2frames(sec, fps):
    return int(sec * fps)


def frames2sec(frames, fps):
    return frames / fps


def get_video_transforms(transforms_config: list) -> torch.nn.Sequential:
    """Returns a torch.nn.Sequential of video transforms according to the config.

    Args:
        transforms_config (list): Config for the transforms.

    Returns:
        torch.nn.Sequential: Transformations to be applied to the video.
    """
    transforms = []
    for transform_config in transforms_config:
        transform = instantiate_from_config(transform_config)
        transforms.append(transform)
    return torch.nn.Sequential(*transforms)


def get_s3d_transforms() -> Tv.Compose:
    return Tv.Compose(
        [
            Tv.Resize(256, antialias=True),
            Tv.CenterCrop((224, 224)),
            Tv.RandomHorizontalFlip(p=0.5),
            Tv.ConvertImageDtype(torch.float32),
        ]
    )


def get_s3d_transforms_validation() -> Tv.Compose:
    return Tv.Compose(
        [
            Tv.Resize(256, antialias=True),
            Tv.CenterCrop((224, 224)),
            Tv.ConvertImageDtype(torch.float32),
        ]
    )


def get_resize_and_convert_to_float32_transforms() -> Tv.Compose:
    return Tv.Compose(
        [
            Tv.Resize(256, antialias=True),
            ToFloat32DType(),
        ]
    )


class ToFloat32DType(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.transform = v2.ConvertDtype(torch.float32)
        except AttributeError:
            self.transform = v2.ToDtype(torch.float32, scale=True)

    def forward(self, x):
        return self.transform(x)


class RandomNullify(torch.nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1) < self.p:
            return x * 0
        else:
            return x


class Permute(torch.nn.Module):
    def __init__(self, permutation: list):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(*self.permutation)


class UniformTemporalSubsample(torch.nn.Module):
    def __init__(self, target_fps: int, clip_duration: float):
        super().__init__()
        self.target_fps = target_fps
        self.clip_duration = clip_duration
        self.temporal_subsampler = v2.UniformTemporalSubsample(
            math.ceil(target_fps * clip_duration)
        )

    def forward(self, x: torch.Tensor):
        return self.temporal_subsampler(x)


class GenerateMultipleSegments(torch.nn.Module):
    """
    Given an item with video and audio, generates a batch of `n_segments` segments
    of length `segment_size_vframes` (if None, the max number of segments will be made).
    If `is_start_random` is True, the starting position of the 1st segment will be random but respecting
    n_segments.
    `audio_jitter_sec` is the amount of audio offset in seconds.
    """

    def __init__(
        self,
        segment_size_vframes: int,
        n_segments: int = None,
        is_start_random: bool = False,
        audio_jitter_sec: float = 0.0,
        step_size_seg: float = 1,
    ):
        super().__init__()
        self.segment_size_vframes = segment_size_vframes
        self.n_segments = n_segments
        self.is_start_random = is_start_random
        self.audio_jitter_sec = audio_jitter_sec
        self.step_size_seg = step_size_seg

    def forward(self, item, segment_a=False):
        v_len_frames, C, H, W = item["video"].shape

        v_fps = int(item["meta"]["video"]["fps"][0])

        ## Determining the number of segments
        # segment size
        segment_size_vframes = self.segment_size_vframes

        # step size (stride)
        stride_vframes = int(self.step_size_seg * segment_size_vframes)

        # calculating the number of segments. (W - F + 2P) / S + 1
        n_segments_max_v = (
            math.floor((v_len_frames - segment_size_vframes) / stride_vframes) + 1
        )

        if segment_a:
            a_len_frames = item["audio"].shape[0]
            a_fps = int(item["meta"]["audio"]["framerate"][0])
            segment_size_aframes = (
                sec2frames(frames2sec(self.segment_size_vframes, v_fps), a_fps)
                if segment_a
                else None
            )
            stride_aframes = (
                int(self.step_size_seg * segment_size_aframes) if segment_a else None
            )
            n_segments_max_a = (
                (math.floor((a_len_frames - segment_size_aframes) / stride_aframes) + 1)
                if segment_a
                else None
            )
            # making sure audio and video can accommodate the same number of segments
            n_segments_max = min(n_segments_max_v, n_segments_max_a)
        else:
            a_len_frames = None
            a_fps = None
            segment_size_aframes = None
            n_segments_max_a = None
            n_segments_max = n_segments_max_v

        n_segments = n_segments_max if self.n_segments is None else self.n_segments

        assert n_segments <= n_segments_max, (
            f"cant make {n_segments} segs of len {self.segment_size_vframes} in a vid "
            f'of len {v_len_frames} for {item["path"]}'
        )

        # (n_segments, 2) each
        v_ranges, a_ranges = self.get_sequential_seg_ranges(
            v_len_frames, a_len_frames, v_fps, a_fps, n_segments, segment_size_aframes
        )

        # segmenting original streams (n_segments, segment_size_frames, C, H, W)
        item["video"] = torch.stack([item["video"][s:e] for s, e in v_ranges], dim=0)
        if segment_a:
            item["audio"] = torch.stack(
                [item["audio"][s:e] for s, e in a_ranges], dim=0
            )
        return item

    def get_sequential_seg_ranges(
        self, v_len_frames, a_len_frames, v_fps, a_fps, n_seg, seg_size_aframes
    ):
        # if is_start_random is True, the starting position of the 1st segment will
        # be random but respecting n_segments like so: "-CCCCCCCC---" (maybe with fixed overlap),
        # else the segments are taken from the middle of the video respecting n_segments: "--CCCCCCCC--"
        segment_a = seg_size_aframes is not None

        seg_size_vframes = self.segment_size_vframes  # for brevity

        # calculating the step size in frames
        step_size_vframes = int(self.step_size_seg * seg_size_vframes)

        # calculating the length of the sequence of segments (and in frames)
        seg_seq_len = n_seg * self.step_size_seg + (1 - self.step_size_seg)
        vframes_seg_seq_len = int(seg_seq_len * seg_size_vframes)

        # doing temporal crop
        max_v_start_i = v_len_frames - vframes_seg_seq_len
        if self.is_start_random:
            v_start_i = random.randint(0, max_v_start_i)
        else:
            v_start_i = max_v_start_i // 2

        # make segments starts
        v_start_seg_i = torch.tensor(
            [v_start_i + i * step_size_vframes for i in range(n_seg)]
        ).int()

        # make segment ends
        v_ends_seg_i = v_start_seg_i + seg_size_vframes

        # make ranges
        v_ranges = torch.stack([v_start_seg_i, v_ends_seg_i], dim=1)
        assert (v_ranges <= v_len_frames).all(), f"{v_ranges} out of {v_len_frames}"

        a_ranges = None
        if segment_a:
            step_size_aframes = int(self.step_size_seg * seg_size_aframes)
            aframes_seg_seq_len = int(seg_seq_len * seg_size_aframes)
            a_start_i = sec2frames(
                frames2sec(v_start_i, v_fps), a_fps
            )  # vid frames -> seconds -> aud frames
            a_start_seg_i = torch.tensor(
                [a_start_i + i * step_size_aframes for i in range(n_seg)]
            ).int()
            # apply jitter to audio
            if self.audio_jitter_sec > 0:
                jitter_aframes = sec2frames(self.audio_jitter_sec, a_fps)
                # making sure after applying jitter, the audio is still within the audio boundaries
                jitter_aframes = min(
                    jitter_aframes,
                    a_start_i,
                    a_len_frames - a_start_i - aframes_seg_seq_len,
                )
                a_start_seg_i += random.randint(
                    -jitter_aframes, jitter_aframes
                )  # applying jitter to segments
            a_ends_seg_i = (
                a_start_seg_i + seg_size_aframes
            )  # using the adjusted a_start_seg_i (with jitter)
            a_ranges = torch.stack([a_start_seg_i, a_ends_seg_i], dim=1)
            assert (a_ranges >= 0).all() and (
                a_ranges <= a_len_frames
            ).all(), f"{a_ranges} out of {a_len_frames}"

        return v_ranges, a_ranges
