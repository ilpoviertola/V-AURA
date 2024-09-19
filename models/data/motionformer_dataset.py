import typing as tp
from pathlib import Path

import torch

from models.data.video_dataset import VideoMeta
from models.data.vjepa_dataset import VJEPADataset
from models.data.transforms.video_transforms import GenerateMultipleSegments


class MotionFormerDataset(VJEPADataset):
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
        """Video dataset designed for the MotionFormer model.

        Args:
            split (str): Which dataset split to use.
            metadata (tp.List[VideoMeta]): Metadata for the dataset.
            sample_duration (float): Duration of single sample. The dataset can consists from videos longer than this duration.
            max_load_attempts (int, optional): Max attempts to load a clip from the dataset. Defaults to 10.
            filter_on_duration (bool, optional): Filter videos with duration < sample_duration from the dataset. Defaults to True.
            discarded_files (tp.List[tp.Union[str, Path]], optional): Exclusively define files to discard. Defaults to [].
            crop (bool, optional): Crop/clip samples to make the frame amounts match. Defaults to True.
            audio_transforms (tp.Optional[torch.nn.Sequential], optional): Audio transformations. Defaults to None.
            video_transforms (tp.Optional[torch.nn.Sequential], optional): Video transformations. Defaults to None.
            partition_audio_to_clips (bool, optional): Wheter to partition audio. Defaults to False.
            partition_video_to_clips (bool, optional): Whether to partition video. Defaults to True.
            frames_per_clip (int, optional): Frame count per video clip. Defaults to 16.
            frame_step (int, optional): Step between the consecutive frames. Defaults to 1.
            model_fps (float, optional): FPS MotionFormer was trained on. Defaults to 25.0.
            assert_fps (bool, optional): Sample FPS must be model_fps. Defaults to True.
        """
        super().__init__(
            split,
            metadata,
            sample_duration,
            max_load_attempts,
            filter_on_duration,
            discarded_files,
            crop,
            audio_transforms,
            video_transforms,
            partition_audio_to_clips,
            partition_video_to_clips,
            frames_per_clip,
            frame_step,
            model_fps,
            assert_fps,
        )

        # MotionFormer related parameters
        del self.jepa_fps
        self.motionformer_fps = model_fps

        if self.partition_audio_to_clips or self.partition_video_to_clips:
            self.to_segments_transform = GenerateMultipleSegments(
                segment_size_vframes=self.frames_per_clip,
                n_segments=-1,  # updated in __getitem__
                is_start_random=(self.split == "train"),
                audio_jitter_sec=0.0,
                step_size_seg=self.frame_step,
            )

    def __getitem__(self, idx) -> dict:
        return super().__getitem__(idx)

    def to_video_segments(self, item: dict) -> dict:
        """Partition video to clips.

        Args:
            item (dict): Single sample.

        Returns:
            dict: Sample with partitioned video.
        """
        num_clips = item["frames"].shape[0] // self.frames_per_clip // self.frame_step
        assert num_clips, "num_clips is zero"
        self.to_segments_transform.n_segments = num_clips

        tmp_meta = {
            "video": {"fps": [item["meta"]["video_fps"]]},
            "audio": {"framerate": [item["meta"]["audio_fps"]]},
        }
        tmp_item = {
            "video": item["frames"],
            "audio": item["audio"].mean(dim=0),
            "path": item["meta"]["filepath"],
            "meta": tmp_meta,
        }
        tmp_item = self.to_segments_transform(tmp_item)

        if self.partition_audio_to_clips:
            item["audio"] = tmp_item["audio"]
        if self.partition_video_to_clips:
            item["frames"] = tmp_item["video"].permute(0, 2, 1, 3, 4)
        return item

    def to_audio_segments(self, item: dict) -> dict:
        # NOTE: this is segmented already in to_video_segments.
        # At the moment there is no such use case where audio is segmented but video is not.
        return item
