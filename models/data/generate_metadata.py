"""Script to generate JSONL metadata files for the datasets."""

import argparse
from pathlib import Path
import json

import ffmpeg
import csv


def get_frame_rate(fps: str):
    if "/" in fps:
        fps = fps.split("/")
        return int(fps[0]) / int(fps[1])
    else:
        return float(fps)


def get_ffmpeg_metadata(fn: str, dir: Path):
    metadata = ffmpeg.probe(str(dir / fn))
    assert len(metadata["streams"]) == 2
    if metadata["streams"][0]["codec_type"] == "video":
        video_stream = metadata["streams"][0]
        audio_stream = metadata["streams"][1]
    else:
        video_stream = metadata["streams"][1]
        audio_stream = metadata["streams"][0]

    return {
        "duration": min(
            float(video_stream["duration"]), float(audio_stream["duration"])
        ),
        "audio_codec_name": audio_stream["codec_name"],
        "audio_fps": int(audio_stream["sample_rate"]),
        "audio_channels": audio_stream["channels"],
        "video_codec_name": video_stream["codec_name"],
        "video_fps": get_frame_rate(video_stream["avg_frame_rate"]),
        "video_width": int(video_stream["width"]),
        "video_height": int(video_stream["height"]),
        "pix_fmt": video_stream["pix_fmt"],
    }


def get_vgg_object(fn: str, dir: Path, desc: str, **kwargs):
    fn = f"{fn}.mp4"

    ffmpeg_metadata = get_ffmpeg_metadata(fn, dir)
    base_metadata = {
        "filepath": (dir / fn).resolve().as_posix(),
        "description": desc,
    }
    base_metadata.update(ffmpeg_metadata)
    return base_metadata


def get_greatesthit_object(
    fn: str, dir: Path, material: str, action_type: str, effect: str, **kwargs
):
    ffmpeg_metadata = get_ffmpeg_metadata(fn, dir)
    base_metadata = {
        "filepath": (dir / fn).resolve().as_posix(),
        "material": material,
        "action_type": action_type,
        "effect": effect,
    }
    base_metadata.update(ffmpeg_metadata)
    return base_metadata


def get_vas_object(fn: str, dir: Path, **kwargs):
    ffmpeg_metadata = get_ffmpeg_metadata(fn, dir)
    base_metadata = {
        "filepath": (dir / fn).resolve().as_posix(),
    }
    base_metadata.update(ffmpeg_metadata)
    return base_metadata


def get_greatesthit_filename(fn: str, datadir: Path, split: str) -> list:
    all_files = (
        datadir.glob(f"{fn}_denoised*")
        if split != "predict"
        else datadir.glob(f"{fn}*")
    )
    return [
        f.name for f in list(all_files) if f.suffix == ".mp4"
    ]  # return only filenames


def get_vgg_filename(fn: str, *args, **kwargs) -> list:
    fn = "_".join(fn.split("_")[:-2])
    return [fn]


def get_vas_filenames(datadir: Path) -> list:
    all_files = datadir.glob("**/*.mp4")
    return [f for f in list(all_files) if f.suffix == ".mp4"]


def get_args():
    parser = argparse.ArgumentParser(description="Generate metadata for the dataset.")
    parser.add_argument(
        "--split_name",
        type=str,
        required=True,
        help="Name of the split.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        nargs="+",
        required=True,
        help="Path to the data directory(ies).",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        nargs="+",
        required=True,
        help="Path to the metadata file(s).",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        nargs="+",
        required=True,
        help="Path to the split file(s).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory.",
    )
    return parser


def main():
    args = get_args().parse_args()
    print(args)

    assert len(args.metadata_file) == len(args.split_file) == len(args.data_path)

    output_dir: Path = Path(args.output_dir) / args.split_name
    output_dir.mkdir(parents=True, exist_ok=False)
    output_file = output_dir / "data.jsonl"
    output_file.touch(exist_ok=False)

    data_object_getters = {
        "vggsound": get_vgg_object,
        "greatesthit": get_greatesthit_object,
        "greatesthits": get_greatesthit_object,
    }

    filename_solvers = {
        "vggsound": get_vgg_filename,
        "greatesthit": get_greatesthit_filename,
        "greatesthits": get_greatesthit_filename,
    }

    for metadata_file, split_file, data_path in zip(
        args.metadata_file, args.split_file, args.data_path
    ):
        metadata_file = Path(metadata_file)
        split_file = Path(split_file)
        data_path = Path(data_path)

        assert metadata_file.exists() and metadata_file.is_file()
        assert split_file.exists() and split_file.is_file()
        assert data_path.exists() and data_path.is_dir()

        if "vggsound" in data_path.as_posix():
            data_object_getter = data_object_getters["vggsound"]
            file_name_solver = filename_solvers["vggsound"]
        elif (
            "greatesthit" in data_path.as_posix()
            or "greatesthits" in data_path.as_posix()
        ):
            data_object_getter = data_object_getters["greatesthit"]
            file_name_solver = filename_solvers["greatesthit"]
        else:
            raise ValueError("Unknown dataset type.")

        with open(split_file, "r") as f:
            split = f.read().splitlines()

        with open(metadata_file, "r") as f:
            metadata_reader = csv.DictReader(f)
            metadata = [row for row in metadata_reader]

        with open(output_file, "a") as f:
            for file in split:
                try:
                    fns = file_name_solver(file, data_path, args.split_name)
                    for fn in fns:
                        m = next((d for d in metadata if d["filename"] == fn), None)
                        if metadata is not None:
                            if "vggsound" in data_path.as_posix():
                                fn = file
                            obj = data_object_getter(fn, data_path, **m)
                            json.dump(obj, f)
                            f.write("\n")
                except Exception as e:
                    print(f"Error processing {fn}: {e}")
                    continue


def generate_vas():
    output_file = Path("data/vas/data.jsonl")
    data = {}
    with open(output_file, "a") as f:
        try:
            for fn in get_vas_filenames(Path("/path/to/VAS")):
                obj = get_vas_object(fn.name, fn.parent)
                json.dump(obj, f)
                f.write("\n")
        except Exception as e:
            print(f"Error processing {fn}: {e}")


if __name__ == "__main__":
    main()
