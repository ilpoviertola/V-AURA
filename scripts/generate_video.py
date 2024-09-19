"""Merge (generated) audio with corresponding video files."""

import sys

sys.path.append(".")  # isort:skip  # noqa: E402
import argparse
import random
from pathlib import Path
import json

import torchaudio
from tqdm import tqdm

from utils.utils import write_video
from utils.data_utils import read_video_to_frames_and_audio_streams


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--samples-dir", "-i", type=str, nargs="+", required=True)
    args.add_argument("--video-dir", "-v", type=str, required=True)
    args.add_argument("--output-path", "-o", type=str, nargs="+", required=False)
    args.add_argument("-vfps", type=float, required=False, default=25)
    args.add_argument("-afps", type=int, required=False, default=24000)
    args.add_argument("--num-videos", type=int, required=False, default=None)
    return args


def main():
    args = get_args().parse_args()
    if args.output_path and len(args.output_path) != len(args.samples_dir):
        raise ValueError(
            "Number of output paths should be equal to number of input paths or none."
        )
    input_paths = [Path(dir) for dir in args.samples_dir]
    video_path = Path(args.video_dir)
    output_paths = (
        [Path(path) for path in args.output_path] if args.output_path else input_paths
    )

    for input_path, output_path in zip(input_paths, output_paths):
        audio_samples = input_path.glob("*.wav")

        if args.num_videos is not None and args.num_videos > 0:
            audio_samples = random.sample(list(audio_samples), args.num_videos)

        for audio in tqdm(
            audio_samples,
            desc=f"Generating videos from {input_path.name}",
            # total=len(list(audio_samples)),
        ):
            metadata = audio.with_suffix(".json")
            if not metadata.exists():
                print("Skipping audio file without metadata", audio.name)
                continue

            metadata = json.load(metadata.open())
            start_pts = metadata["conditioning"]["video"]["seek_time"][0]
            video_name = Path(metadata["conditioning"]["video"]["path"][0]).name

            audio_tensor, sr = torchaudio.load(audio, channels_first=True)
            duration = audio_tensor.size(1) / sr

            frames, _, _ = read_video_to_frames_and_audio_streams(
                (video_path / video_name).as_posix(),
                start_pts=start_pts,
                end_pts=start_pts + duration,
            )

            write_video(
                filename=(output_path / audio.with_suffix(".mp4").name).as_posix(),
                video_array=frames.permute(0, 2, 3, 1),
                audio_array=audio_tensor,
                fps=args.vfps,
                video_codec="h264",
                options={"crf": "10", "pix_fmt": "yuv420p"},
                audio_fps=args.afps,
                audio_codec="aac",
            )


if __name__ == "__main__":
    main()
