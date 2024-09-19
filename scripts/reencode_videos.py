"""
Adapted from https://github.com/v-iashin/SparseSync/blob/main/scripts/reencode_videos.py.

Use to reencode videos to a specific format. See README.md for more details.
"""

import subprocess
from glob import glob
from multiprocessing import Pool
from pathlib import Path
import random

from tqdm import tqdm

ORIG_PATH = Path("")  # Path to the folder with videos to reencode
NUM_WORKERS = 32

# V-AURA defaults
V_FPS = 25  # 25 fps is the V-AURA default
MIN_SIDE = 256  # 256 is the V-AURA default
A_FPS = 44100  # 44100 Hz is the V-AURA default
VCODEC = "h264"  # h264 is the V-AURA default
CRF = 10  # 10 is the V-AURA default
PIX_FMT = "yuv420p"  # yuv420p is the V-AURA default
ACODEC = "aac"  # aac is the V-AURA default


def which_ffmpeg() -> str:
    """Determines the path to ffmpeg library
    Returns:
        str -- path to the library
    """
    result = subprocess.run(
        ["which", "ffmpeg"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    ffmpeg_path = result.stdout.decode("utf-8").replace("\n", "")
    return ffmpeg_path


def get_new_path(
    path, vcodec, acodec, v_fps, min_side, a_fps, orig_path, prefix="video"
) -> Path:
    new_folder_name = f"{vcodec}_{prefix}_{v_fps}fps_{min_side}side_{a_fps}hz_{acodec}"
    if "vggsound" in str(orig_path):
        new_folder_path = orig_path.parents[1] / new_folder_name
    elif (
        "mjpeg" in str(orig_path)
        or "lrs3" in str(orig_path)
        or "audioset" in str(orig_path)
    ):
        new_folder_path = Path(
            str(path.parent).replace(orig_path.name, f"/{new_folder_name}/")
        )
    elif "greatesthit" in str(orig_path):
        new_folder_path = orig_path.parents[0] / new_folder_name
    else:
        raise NotImplementedError
    new_folder_path.mkdir(exist_ok=True, parents=True)
    new_path = new_folder_path / path.name
    return new_path


def reencode_video(path):
    new_path = get_new_path(path, VCODEC, ACODEC, V_FPS, MIN_SIDE, A_FPS, ORIG_PATH)
    # reencode the original mp4: rescale, resample video and resample audio
    cmd = f"{which_ffmpeg()}"
    # no info/error printing
    cmd += " -hide_banner -loglevel panic"
    cmd += f" -i {path}"
    # 1) change fps, 2) resize: min(H,W)=MIN_SIDE (vertical vids are supported), 3) change audio framerate
    cmd += f" -vf fps={V_FPS},scale=iw*{MIN_SIDE}/'min(iw,ih)':ih*{MIN_SIDE}/'min(iw,ih)',crop='trunc(iw/2)'*2:'trunc(ih/2)'*2"
    cmd += f" -vcodec {VCODEC} -pix_fmt {PIX_FMT} -crf {CRF}"
    cmd += f" -acodec {ACODEC} -ar {A_FPS} -ac 1"
    cmd += f" {new_path}"
    if new_path.exists():
        print("already exists", new_path)
    else:
        subprocess.call(cmd.split())


def main():
    assert (
        which_ffmpeg() != ""
    ), "Is ffmpeg installed? Check if the conda environment is activated."

    if ORIG_PATH.is_file():
        video_paths = [ORIG_PATH]
    else:
        if "vggsound" in str(ORIG_PATH):
            paths_glob = str(ORIG_PATH / "*.mp4")
        elif "lrs3" in str(ORIG_PATH):
            paths_glob = str(ORIG_PATH / "*/*/*.mp4")
        elif "greatesthit" in str(ORIG_PATH):
            paths_glob = str(ORIG_PATH / "*_denoised.mp4")
        elif "audioset" in str(ORIG_PATH):
            paths_glob = str(ORIG_PATH / "*/*.mp4")
        video_paths = [Path(p) for p in sorted(glob(paths_glob))]
    print(len(video_paths))
    assert len(video_paths) > 0

    random.shuffle(video_paths)
    reencode_fn = reencode_video

    # single thread (slow)
    # for path in tqdm(video_paths):
    #     reencode_fn(path)

    # multiple threads (fast)
    with Pool(NUM_WORKERS) as pool:
        list(tqdm(pool.imap(reencode_fn, video_paths), total=len(video_paths)))

    print(f"{VCODEC}_video_{V_FPS}fps_{MIN_SIDE}side_{A_FPS}hz_{ACODEC}")


if __name__ == "__main__":
    main()
