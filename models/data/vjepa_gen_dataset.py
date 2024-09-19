import typing as tp
from pathlib import Path
import csv
import warnings

from models.data.video_dataset import VideoMeta
from models.data.vjepa_dataset import VJEPADataset


class VJEPAGenDataset(VJEPADataset):
    def __init__(self, gen_videos_filepath: str, *args, **kwargs):
        """VJEPA dataset when generating videos.

        Args:
            gen_videos_filepath (Path): Path to a CSV file containing the generated video filenames and segment start secs.
        """
        if gen_videos_filepath is None or not Path(gen_videos_filepath).is_file():
            warnings.warn(
                f"File {gen_videos_filepath} does not exist. Sampled start points will be 0."
            )
            self.gen_videos_filepath = None
        else:
            self.gen_videos_filepath = Path(gen_videos_filepath)
        self.start_pts: tp.Dict[str, float] = {}
        super().__init__(*args, **kwargs)

    def _init_dataset(
        self,
        metadata: tp.List[VideoMeta],
        filter_on_duration: bool = True,
        discarded_files: tp.List[tp.Union[str, Path]] = [],
    ) -> tp.List[VideoMeta]:
        if filter_on_duration:
            metadata = [
                m for m in metadata if m.duration >= self.sample_duration + self.EPS
            ]

        if self.gen_videos_filepath is None:
            return metadata

        included_filenames = set()
        with open(self.gen_videos_filepath, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                filename, start_sec = row
                self.start_pts[filename] = float(start_sec)
                included_filenames.add(filename)

        metadata = [m for m in metadata if Path(m.filepath).stem in included_filenames]
        return metadata

    def _sample_start_pts(self, idx, sample_duration: float, video_len: float) -> float:
        return self.start_pts.get(Path(self.dataset[idx].filepath).stem, 0.0)
