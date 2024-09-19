"""
This script will only cut the video and audio streams into 2 * gh_action_buffer (± gh_action_buffer_noise) seconds
clips, and save these clips into output-path as MPEG4 files.

This script will not resample the video or audio streams! If you need to do that, use
the script in scripts/reencode_videos.py.
"""

import sys
import json
import csv
from pathlib import Path
from argparse import ArgumentParser
from math import ceil, floor
from multiprocessing import Pool, cpu_count
import typing as tp
from fractions import Fraction

import numpy as np
from pytorchvideo.data.encoded_video import EncodedVideo

sys.path.append(".")
from utils.utils import write_video

SEGMENTING_TACTIC_TYPES = ["dummy", "annotations", "random"]
GREATEST_HIT_ACTION_TYPES = ["scratch", "hit"]
EPS = 1e-9
VFPS = 25  # original greatest hit videos are 30 fps
AFPS = 24000  # original greatest hit videos are 96k Hz
LONGEST_VIDEO_CLIP_IN_SAMPLES = -1
LONGEST_AUDIO_CLIP_IN_SAMPLES = -1


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument(
        "--split_tactic", type=str, required=True, choices=SEGMENTING_TACTIC_TYPES
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--filename_mask", type=str, default="*_denoised.mp4")
    parser.add_argument("--gh_meta_filename_mask", type=str, default="*_times.txt")
    parser.add_argument("--gh_action_buffer", type=float, default=0.5)
    parser.add_argument("--gh_action_buffer_noise", type=float, default=None)
    return parser.parse_args()


def get_input_files(input_path: str, filename_mask: str) -> tp.List[Path]:
    inp = Path(input_path)
    if inp.is_file():
        files2process = sorted(list(inp.parent.glob(filename_mask)))
    else:
        files2process = sorted(list(inp.glob(filename_mask)))
    return files2process


def init_output_folder(output_path: str, split_tactic: str, buffer_len: float) -> Path:
    outp = Path(output_path)
    outp = outp.parent / (
        outp.name + f"_len_{int(buffer_len * 2)}" + f"_splitby_{split_tactic}"
    )
    outp.mkdir(exist_ok=True, parents=True)
    return outp


def check_video_and_meta_files(
    video_files: tp.List[Path], meta_files: tp.List[Path], split_tactic: str
) -> tp.Tuple[tp.List[Path], tp.List[tp.Union[Path, None]]]:
    # TODO: Check that the order of files is the same
    if split_tactic == "annotations":
        assert len(video_files) == len(meta_files), (
            "The number of videos and the number of metadata files do not match!"
            f" {len(video_files)} != {len(meta_files)}"
        )
        return video_files, meta_files
    else:
        if len(video_files) != len(meta_files):
            print(
                "The number of videos and the number of metadata files do not match!"
                f" {len(video_files)} != {len(meta_files)}"
            )
            print("Discarding meta files.")
            return video_files, [None] * len(video_files)
        else:
            return video_files, meta_files


def is_close(a: float, b: list, eps: float = EPS) -> bool:
    return any(abs(a - x) < eps for x in b)


def closest_metadata_line(
    occurring_time: float,
    metadata_path: tp.Optional[Path] = None,
) -> str:
    if metadata_path is None:
        return f"{str(occurring_time)} None None None"
    else:
        with open(metadata_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        annotation_ts = [float(line.split(" ")[0]) for line in lines]
        # get closest timestamp from annotations respect to the randomly/uniformly sampled one
        closest_ts = min(
            annotation_ts,
            key=lambda x: abs(x - occurring_time),
        )
        closest_line = lines[annotation_ts.index(closest_ts)].split(" ")
        closest_line[0] = str(occurring_time)
        return " ".join(closest_line)


def add_noise(lines: tp.List[str], buffer_noise: float) -> tp.List[str]:
    # add noise to the buffer
    lines = [
        line.replace(
            line.split(" ")[0],
            str(
                round(
                    np.random.normal(
                        float(line.split(" ")[0]),
                        buffer_noise,
                    ),
                    3,
                )
            ),
        )
        for line in lines
    ]
    return lines


def get_samples(
    split_tactic: str,
    gh_action_buffer: float,
    video: EncodedVideo,
    buffer_noise: tp.Optional[float],
    meta_path: tp.Optional[Path],
    max_tries: int = 313,
) -> tp.List[str]:
    # split according to annotations (old tactic)
    if split_tactic == "annotations":
        assert (
            meta_path is not None
        ), "No metadata file provided for annotations based split!"
        with open(meta_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    # split uniformly
    elif split_tactic == "dummy":
        single_clip_length = gh_action_buffer * 2
        clip_lengths = np.linspace(
            0,
            video.duration,
            int(video.duration // single_clip_length),
            endpoint=False,
        )
        lines = [
            closest_metadata_line(start_time, meta_path) for start_time in clip_lengths
        ]
    # split randomly
    elif split_tactic == "random":
        lines = []
        sampled_ts: tp.List[float] = []
        sample_count = floor(video.duration)
        i = 0
        while len(lines) < sample_count:
            occurring_time = np.random.uniform(
                0 + gh_action_buffer + EPS,
                float(video.duration) - gh_action_buffer - EPS,
            )
            occurring_time = round(occurring_time, 3)
            # if occurring time is too close to the already sampled ones, skip
            if is_close(occurring_time, sampled_ts, 0.5):
                if i > max_tries:
                    print(
                        f"Could not sample {sample_count} times with the given buffer length {gh_action_buffer}. Got {len(sampled_ts)} samples. Exiting."
                    )
                    break
                i += 1
                continue
            sampled_ts.append(occurring_time)
            lines.append(closest_metadata_line(occurring_time, meta_path))
            i = 0  # reset the counter
    else:
        raise NotImplementedError(f"Unknown split tactic {split_tactic}")

    if buffer_noise is not None:
        lines = add_noise(lines, buffer_noise)

    return lines


def process_video(
    video_path: Path,
    output_path: Path,
    gh_action_buffer: float,
    buffer_noise: float,
    split_tactic: str,
    meta_path: tp.Optional[Path] = None,
) -> tp.List[tp.List[str]]:
    global LONGEST_VIDEO_CLIP_IN_SAMPLES, LONGEST_AUDIO_CLIP_IN_SAMPLES
    video = EncodedVideo.from_path(video_path)
    assert (
        round(video._container.streams.video[0].guessed_rate) == VFPS
    ), f"Video FPS is not {VFPS}!"
    assert video._audio_time_base == Fraction(
        1, AFPS
    ), f"Audio sample rate is not {AFPS}!"
    metadatas: tp.List[tp.List[str]] = []
    sample_metas = get_samples(
        split_tactic, gh_action_buffer, video, buffer_noise, meta_path
    )

    for meta in sample_metas:
        occuring_time, material, action_type, effect = meta.split(" ")
        # TODO: think about this... maybe we do not care even if the annotation is not sensible?
        # if action_type.lower() not in GREATEST_HIT_ACTION_TYPES:
        #     continue

        occuring_ts = float(occuring_time)
        start_time = round(max(0, occuring_ts - gh_action_buffer - EPS), 3)
        end_time = round(
            min(float(video.duration), occuring_ts + gh_action_buffer + EPS), 3
        )
        id = ceil(occuring_ts * VFPS)
        name = f"{video_path.stem}_{id}.mp4"

        clip = video.get_clip(start_time, end_time)

        LONGEST_VIDEO_CLIP_IN_SAMPLES = max(
            LONGEST_VIDEO_CLIP_IN_SAMPLES, clip["video"].shape[1]
        )

        clip["audio"] = clip["audio"].unsqueeze(0)
        LONGEST_AUDIO_CLIP_IN_SAMPLES = max(
            LONGEST_AUDIO_CLIP_IN_SAMPLES, clip["audio"].shape[1]
        )
        write_video(
            filename=(Path(output_path) / name).as_posix(),
            video_array=clip["video"].permute(1, 2, 3, 0),
            fps=VFPS,
            video_codec="h264",
            options={"crf": "10", "pix_fmt": "yuv420p"},
            audio_array=clip["audio"],
            audio_fps=AFPS,
            audio_codec="aac",
        )

        metadatas.append(
            [
                name,
                str(start_time),
                occuring_time,
                str(end_time),
                material,
                action_type,
                effect,
            ]
        )

    video.close()
    return metadatas


if __name__ == "__main__":
    args = get_args()

    videos2process = get_input_files(args.input, args.filename_mask)
    assert len(videos2process) > 0, "No videos found!"

    metafiles2process = get_input_files(args.input, args.gh_meta_filename_mask)
    videos2process, metafiles2process = check_video_and_meta_files(
        videos2process, metafiles2process, args.split_tactic
    )

    if args.output is None:
        args.output = args.input
    output_path = init_output_folder(
        args.output, args.split_tactic, args.gh_action_buffer
    )

    metadata = []

    with Pool(cpu_count()) as p:
        metadata += p.starmap(
            process_video,
            [
                (
                    video,
                    output_path,
                    args.gh_action_buffer,
                    args.gh_action_buffer_noise,
                    args.split_tactic,
                    meta,
                )
                for video, meta in zip(videos2process, metafiles2process)
            ],
        )

    # fix metadata formatting
    metadata_f = [item for sublist in metadata for item in sublist]
    # metadata_f = metadata
    metadata_f.insert(
        0,
        [
            "filename",
            "start_time",
            "occurring_time",
            "end_time",
            "material",
            "action_type",
            "effect",
        ],
    )

    with open(output_path / "metadata.csv", "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(metadata_f)
    with open(output_path / "longest_clip_in_samples.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "video": LONGEST_VIDEO_CLIP_IN_SAMPLES,
                "audio": LONGEST_AUDIO_CLIP_IN_SAMPLES,
            },
            f,
        )
    print("saved to", output_path)
