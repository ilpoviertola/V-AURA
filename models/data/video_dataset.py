import typing as tp
from pathlib import Path
import logging
from fractions import Fraction
from dataclasses import dataclass, fields
import gzip
import json

import torch
from torch.utils.data import Dataset
import numpy as np

from utils.data_utils import read_video_to_frames_and_audio_streams_with_av


logger = logging.getLogger(__name__)


@dataclass(order=True)
class BaseInfo:

    @classmethod
    def _dict2fields(cls, dictionary: dict):
        return {
            field.name: dictionary[field.name]
            for field in fields(cls)
            if field.name in dictionary
        }

    @classmethod
    def from_dict(cls, dictionary: dict):
        _dictionary = cls._dict2fields(dictionary)
        return cls(**_dictionary)

    def to_dict(self):
        return {field.name: self.__getattribute__(field.name) for field in fields(self)}


@dataclass(order=True)
class VideoMeta(BaseInfo):
    filepath: str
    duration: float
    audio_codec_name: str
    audio_fps: int
    audio_channels: int
    video_codec_name: str
    video_fps: float
    video_width: int
    video_height: int
    pix_fmt: str
    # additional data fields
    description: str = ""
    material: str = ""
    action_type: str = ""
    effect: str = ""

    @classmethod
    def from_dict(cls, dictionary: dict):
        base = cls._dict2fields(dictionary)
        return cls(**base)

    def to_dict(self):
        d = super().to_dict()
        return d


def load_video_meta(
    path: tp.Union[str, Path], resolve: bool = False
) -> tp.List[VideoMeta]:
    """Load list of Video from an optionally compressed json file.

    Args:
        path (str or Path): Path to JSON file.
        resolve (bool): Whether to resolve the path from Video (default=False).
    Returns:
        list of VideoMeta: List of video file path, its total duration and other metadata.
    """
    open_fn = gzip.open if str(path).lower().endswith(".gz") else open
    with open_fn(path, "rb") as fp:  # type: ignore
        lines = fp.readlines()
    meta = []
    for line in lines:
        d = json.loads(line)
        m = VideoMeta.from_dict(d)
        if resolve:
            m.filepath = Path(m.filepath).resolve().as_posix()
        meta.append(m)
    return meta


class VideoDataset(Dataset):
    """Baseclass for video datasets."""

    EPS = torch.finfo(torch.float32).eps

    def __init__(
        self,
        split: str,
        metadata: tp.List[VideoMeta],
        sample_duration: float,
        max_load_attempts: int = 10,
        filter_on_duration: bool = True,
        discarded_files: tp.List[tp.Union[str, Path]] = [],
        crop: bool = True,
    ):
        """Initialize the dataset.

        Args:
            split (str): The split of the dataset. One of "train", "validation", "test", "predict".
            metadata (tp.List[VideoMeta]): A list of VideoMeta objects.
            sample_duration (float): The duration of the sample in seconds.
            max_load_attempts (int): The maximum number of attempts to load a video.
            filter_on_duration (bool): Whether to filter out videos that are shorter than the sample duration.
            discarded_files (tp.List[tp.Union[str, Path]]): A list of filenames, path to files containing the filenames,
                or paths to directories containing the files with filenames to discard from the dataset.
            crop (bool): Whether to crop the video and audio to the sample duration.
        """
        assert max_load_attempts > 0, "max_load_attempts must be greater than 0."
        assert sample_duration > 0, "sample_duration must be greater than 0."
        self.split = split

        self.sample_duration = sample_duration
        self.max_load_attempts = max_load_attempts
        self.crop = crop

        self.dataset: tp.List[VideoMeta] = self._init_dataset(
            metadata, filter_on_duration, discarded_files
        )
        logger.info("Initialized dataset for split %s with %d files.", split, len(self))

    def _init_dataset(
        self,
        metadata: tp.List[VideoMeta],
        filter_on_duration: bool = True,
        discarded_files: tp.List[tp.Union[str, Path]] = [],
    ) -> tp.List[VideoMeta]:
        """Initialize the dataset."""
        initial_len = len(metadata)
        if filter_on_duration:
            metadata = [
                m for m in metadata if m.duration >= self.sample_duration + self.EPS
            ]
        if discarded_files:
            discarded_filenames = self._solve_discarded_filenames(discarded_files)
            metadata = [
                m for m in metadata if Path(m.filepath).name not in discarded_filenames
            ]
        filtered_len = len(metadata)
        if initial_len != filtered_len:
            logger.info("Filtered out %d files.", initial_len - filtered_len)
        return metadata

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx) -> dict:
        """Return the item at the given index."""
        return self._video_loading_handler(idx)

    def _video_loading_handler(self, idx) -> dict:
        """Handle the loading of the video.

        Args:
            idx (tp.Any): Dataset index.

        Raises:
            RuntimeError: If the video could not be loaded correctly.

        Returns:
            dict: Dictionary containing the frames (THWC), audio and metadata.
        """
        loaded_video = False
        load_attempts = 0
        while not loaded_video and load_attempts < self.max_load_attempts:
            meta = self.dataset[idx]
            if not meta:
                logger.warning("No metadata found for idx %s.", str(idx))
                idx = np.random.randint(0, len(self))
                continue

            data_path = meta.filepath
            start_pts = self._sample_start_pts(idx, self.sample_duration, meta.duration)

            try:
                rgb, audio, loaded_meta = self.load_video_from_file(
                    Path(data_path),
                    meta.video_fps,
                    v_start_s=start_pts,
                    v_end_s=self.sample_duration + start_pts + self.EPS,
                )
                loaded_video = self._video_loaded(
                    rgb.shape, audio.shape, meta.video_fps, meta.audio_fps
                )
            except Exception as e:
                logger.error(e)
                loaded_video = False
                rgb, audio, meta = None, None, None

            if not loaded_video:
                logger.warning(
                    "Video %s could not be loaded correctly. Trying another one.",
                    Path(data_path).name,
                )
                idx = np.random.randint(0, len(self))
                load_attempts += 1

        if not loaded_video:
            raise RuntimeError(
                f"Video could not be loaded correctly. Tried {self.max_load_attempts} times."
            )

        if self.crop:
            rgb = rgb[: int(meta.video_fps * self.sample_duration), ...]
            audio = audio[..., : int(meta.audio_fps * self.sample_duration)]

        meta = meta.to_dict()
        meta["start_pts"] = start_pts
        meta["sample_duration"] = self.sample_duration
        if type(loaded_meta) == dict and type(meta) == dict:
            meta.update(loaded_meta)

        return {"frames": rgb, "audio": audio, "meta": meta}

    def _video_loaded(
        self, rgb_shape: tuple, audio_shape: tuple, vfps: float, afps: float
    ) -> bool:
        """Check if the video was loaded correctly."""
        if rgb_shape is None or audio_shape is None:
            return False
        if rgb_shape[0] < int(vfps * self.sample_duration):
            return False
        if audio_shape[-1] < int(afps * self.sample_duration):
            return False
        return True

    def _sample_start_pts(self, idx, sample_duration: float, video_len: float) -> float:
        """Sample the starting point for the video.

        Args:
            idx: Dataset index
            sample_duration (float): Lenght of the sample.
            video_len (float): Total length of the video.

        Returns:
            float: Starting point for the sample within the video.
        """
        if self.split != "train":
            return 0.0
        return np.random.uniform(0, video_len - sample_duration - self.EPS)

    @staticmethod
    def _solve_discarded_filenames(
        file_list: tp.List[tp.Union[str, Path]]
    ) -> tp.List[str]:
        """Get the filenames of excluded files

        Args:
            file_list (tp.List[str]): List of files, filenames, or directories

        Returns:
            tp.List[str]: List of discarded filenames.
        """

        def filename_getter(f: tp.Union[str, Path]):
            fn_list = []
            with open(f, encoding="utf-8") as open_f:
                discarded_filenames = set(open_f.read().splitlines())
                for filename in discarded_filenames:
                    fn_list.append(Path(filename).with_suffix(".mp4").name)
            return fn_list

        ret = []
        for f in file_list:
            f = Path(f)
            if f.suffix == ".mp4":
                ret.append(f.name)
            elif f.is_file():
                ret.extend(filename_getter(f))
            elif f.is_dir():
                for file in f.iterdir():
                    ret.extend(filename_getter(file))
        return ret

    @staticmethod
    def load_video_from_file(
        file: Path,
        vfps: float,
        v_start_s: tp.Union[float, Fraction] = 0,
        v_end_s: tp.Optional[tp.Union[float, Fraction]] = None,
        a_start_s: tp.Union[float, Fraction] = 0,
        a_end_s: tp.Optional[tp.Union[float, Fraction]] = None,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, tp.Dict[str, tp.Any]]:
        """Load a video from a file.

        Args:
            file (Path): Filepath to the video.
            vfps (float): The frames per second of the video.
            v_start_s (tp.Union[float, Fraction], optional): Start point (in seconds) for the RGB stream. Defaults to 0.
            v_end_s (tp.Optional[tp.Union[float, Fraction]], optional): End point (in seconds) for the RGB stream. Defaults to None.
            a_start_s (tp.Union[float, Fraction], optional): Start point (in seconds) for the audio stream. Defaults to 0.
            a_end_s (tp.Optional[tp.Union[float, Fraction]], optional): End point (in seconds) for the audio stream. Defaults to None.

        Raises:
            ValueError: If video start time is greater than video end time.
            ValueError: If audio start time is greater than audio end time.

        Returns:
            tp.Tuple[torch.Tensor, torch.Tensor, tp.Dict[str, tp.Any]]: RGB (THWC), audio and metadata.
        """
        if v_start_s is not None and a_start_s is None:
            a_start_s = v_start_s
        if a_start_s is not None and v_start_s is None:
            v_start_s = a_start_s
        if v_end_s is not None and a_end_s is None:
            a_end_s = v_end_s
        if a_end_s is not None and v_end_s is None:
            v_end_s = a_end_s

        if v_start_s and v_end_s and v_start_s >= v_end_s:
            raise ValueError(
                f"Invalid video start and end time: {v_start_s} >= {v_end_s}"
            )
        if a_start_s and a_end_s and a_start_s >= a_end_s:
            raise ValueError(
                f"Invalid audio start and end time: {a_start_s} >= {a_end_s}"
            )

        return read_video_to_frames_and_audio_streams_with_av(
            file, vfps, v_start_s, v_end_s, a_start_s, a_end_s
        )

    @classmethod
    def from_meta_file(cls, path: tp.Union[str, Path], **kwargs):
        """Create a VideoDataset from a metadata file.

        Args:
            path (tp.Union[str, Path]): Path to the metadata file.

        Returns:
            VideoDataset: VideoDataset object.
        """
        path = Path(path)
        if path.is_dir():
            if (path / "data.jsonl").exists():
                path = path / "data.jsonl"
            elif (path / "data.jsonl.gz").exists():
                path = path / "data.jsonl.gz"
            else:
                raise ValueError(
                    "Don't know where to read metadata from in the dir. "
                    "Expecting either a data.jsonl or data.jsonl.gz file but none found."
                )
        metadata = load_video_meta(path)
        return cls(metadata=metadata, **kwargs)
