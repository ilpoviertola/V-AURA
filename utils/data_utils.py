from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any, List
from fractions import Fraction
from math import ceil, floor
import warnings
import subprocess as sp
import sys

import torchaudio
from torch import Tensor
from torchvision.io import read_video
from torchvision.io import _video_opt
from torchvision.io.video import (
    _check_av_available,
    _read_from_stream,
    _align_audio_frames,
)
import torch
import av
import numpy as np


def read_video_to_frames_and_audio_streams(
    fn: str,
    start_pts: Optional[Union[float, Fraction]] = 0,
    end_pts: Optional[Union[float, Fraction]] = None,
    pts_unit: str = "sec",
    output_format: str = "TCHW",
) -> Tuple[Tensor, Tensor, dict]:
    """Read video file to frames and audio streams.

    Args:
        fn (str): Path to video file.
        start_pts (Optional[Union[float, Fraction]], optional): Where to start reading video from. Defaults to 0.
        end_pts (Optional[Union[float, Fraction]], optional): Where to end reading video to. Defaults to None.
        pts_unit (str, optional): Unit of measurement. Defaults to "sec".
        output_format (str, optional): Output dim order. Defaults to "TCHW".

    Raises:
        FileNotFoundError: If video does not exist.

    Returns:
        (Tensor, Tensor, dict): Frames (Tv, C, H, W), audio streams (Ta, ), and metadata.
    """
    if not Path(fn).is_file():
        # warnings.warn(f"File {fn} does not exist.")
        return torch.empty(0), torch.empty(0), {}

    frames, audio, metadata = read_video(
        fn, start_pts, end_pts, pts_unit, output_format
    )
    audio = audio.mean(dim=0)  # (2, T) -> (T,)
    return frames, audio, metadata


def loadvideo_decord(
    sample: str,
    frames_per_clip: int,
    frame_step: int,
    num_clips: int,
    allow_clip_overlap: bool = True,
    random_clip_sampling: bool = False,
    filter_short_videos: bool = True,
    duration: Union[float, None] = None,
    filter_long_videos: int = int(10**9),
):
    from decord import VideoReader, cpu

    """Load video content using Decord"""

    fname = sample
    if not Path(fname).exists():
        warnings.warn(f"video path not found {fname=}")
        return [], None

    _fsize = Path(fname).stat().st_size
    if _fsize < 1 * 1024:  # avoid hanging issue
        warnings.warn(f"video too short {fname=}")
        return [], None
    if _fsize > filter_long_videos:
        warnings.warn(f"skipping long video of size {_fsize=} (bytes)")
        return [], None

    try:
        vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))
    except Exception:
        return [], None

    fpc = frames_per_clip
    fstp = frame_step
    if duration is not None:
        try:
            fps = vr.get_avg_fps()
            fstp = int(duration * fps / fpc)
        except Exception as e:
            warnings.warn(str(e))
    clip_len = int(fpc * fstp)

    if filter_short_videos and len(vr) < clip_len:
        warnings.warn(f"skipping video of length {len(vr)}")
        return [], None

    vr.seek(0)  # Go to start of video before sampling frames

    # Partition video into equal sized segments and sample each clip
    # from a different segment
    partition_len = len(vr) // num_clips

    all_indices, clip_indices = [], []
    for i in range(num_clips):

        if partition_len > clip_len:
            # If partition_len > clip len, then sample a random window of
            # clip_len frames within the segment
            end_indx = clip_len
            if random_clip_sampling:
                end_indx = np.random.randint(clip_len, partition_len)
            start_indx = end_indx - clip_len
            indices = np.linspace(start_indx, end_indx, num=fpc)
            indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
            # --
            indices = indices + i * partition_len
        else:
            # If partition overlap not allowed and partition_len < clip_len
            # then repeatedly append the last frame in the segment until
            # we reach the desired clip length
            if not allow_clip_overlap:
                indices = np.linspace(0, partition_len, num=partition_len // fstp)
                indices = np.concatenate(
                    (
                        indices,
                        np.ones(fpc - partition_len // fstp) * partition_len,
                    )
                )
                indices = np.clip(indices, 0, partition_len - 1).astype(np.int64)
                # --
                indices = indices + i * partition_len

            # If partition overlap is allowed and partition_len < clip_len
            # then start_indx of segment i+1 will lie within segment i
            else:
                sample_len = min(clip_len, len(vr)) - 1
                indices = np.linspace(0, sample_len, num=sample_len // fstp)
                indices = np.concatenate(
                    (
                        indices,
                        np.ones(fpc - sample_len // fstp) * sample_len,
                    )
                )
                indices = np.clip(indices, 0, sample_len - 1).astype(np.int64)
                # --
                clip_step = 0
                if len(vr) > clip_len:
                    clip_step = (len(vr) - clip_len) // (num_clips - 1)
                indices = indices + i * clip_step

        clip_indices.append(indices)
        all_indices.extend(list(indices))

    buffer = vr.get_batch(all_indices).asnumpy()
    return buffer, clip_indices


def crop_or_pad_tensor(tensor: Tensor, target_length: int) -> Tensor:
    """
    Crop or pad a 1D PyTorch tensor to the specified target length.

    Args:
        tensor (torch.Tensor): The input 1D tensor.
        target_length (int): The desired length for the output tensor.

    Returns:
        torch.Tensor: The cropped or padded tensor.
    """
    current_length = tensor.size(0)

    if current_length > target_length:
        # If the input tensor is longer than the target length, crop it
        result = tensor[:target_length]
    else:
        # If the input tensor is shorter than the target length, pad it with zeros
        padding = target_length - current_length
        pad_tensor = torch.zeros(padding, dtype=tensor.dtype)
        result = torch.cat((tensor, pad_tensor))

    return result


def read_video_to_frames_and_audio_streams_with_av(
    path: Path,
    vfps: float,
    v_start_s: Union[float, Fraction] = 0,
    v_end_s: Optional[Union[float, Fraction]] = None,
    a_start_s: Optional[Union[float, Fraction]] = None,
    a_end_s: Optional[Union[float, Fraction]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    with av.open(path.as_posix(), metadata_errors="ignore") as av_container:
        # get duration of the video (min of video and audio)
        video = av_container.streams.video[0]
        audio = av_container.streams.audio[0]
        v_duration = float(video.duration * video.time_base)
        a_duration = float(audio.duration * audio.time_base)
        duration_s = min(v_duration, a_duration)
        # v_start_s, v_end_s, a_start_s, a_end_s are defined somewhere here based on randomness and duration_s
        # load video and audio; NOTE: allowing to load a bit more audio (to prevent aud < crop_len_s)
        a_end_s = a_end_s + 2 / vfps if a_end_s is not None else None
        rgb, audio, meta = parse_av_container(
            av_container,
            v_start_s,
            v_end_s,
            a_start_s,
            a_end_s,
            pts_unit="sec",
            output_format="TCHW",
        )
        meta["duration"] = duration_s
    return rgb, audio, meta


def parse_av_container(
    container,
    start_pts: Union[float, Fraction] = 0,
    end_pts: Optional[Union[float, Fraction]] = None,
    audio_start_pts: Optional[Union[float, Fraction]] = None,
    audio_end_pts: Optional[Union[float, Fraction]] = None,
    pts_unit: str = "pts",
    output_format: str = "THWC",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Reads a video from a file, returning both the video frames and the audio frames.
    Extended from https://pytorch.org/vision/main/generated/torchvision.io.read_video.html
        to parameterise the audio start and end pts.

    Args:
        container (TODO type): opened container
        start_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The start presentation time of the video
        end_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The end presentation time
        audio_start_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The start presentation time of the audio
        audio_end_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The end presentation time of the audio
        pts_unit (str, optional): unit in which start_pts and end_pts values will be interpreted,
            either 'pts' or 'sec'. Defaults to 'pts'.
        output_format (str, optional): The format of the output video tensors.
                                       Can be either "THWC" (default) or "TCHW".

    Returns:
        vframes (Tensor[T, H, W, C] or Tensor[T, C, H, W]): the `T` video frames
        aframes (Tensor[K, L]): the audio frames, where `K` is the number of channels and `L` is the number of points
        info (Dict): metadata for the video and audio. Can contain the fields video_fps (float) and audio_fps (int)
    """
    from torchvision import get_video_backend

    _check_av_available()

    if end_pts is None:
        end_pts = float("inf")

    if audio_start_pts is None:
        audio_start_pts = start_pts
    if audio_end_pts is None:
        audio_end_pts = end_pts

    if end_pts < start_pts:
        raise ValueError(
            f"end_pts should be > than start_pts, got start={start_pts} and end={end_pts}"
        )

    info = {}
    video_frames = []
    audio_frames = []
    audio_timebase = _video_opt.default_timebase

    if container.streams.audio:
        audio_timebase = container.streams.audio[0].time_base
    if container.streams.video:
        video_frames = _read_from_stream(
            container,
            start_pts,
            end_pts,
            pts_unit,
            container.streams.video[0],
            {"video": 0},
        )
        video_fps = container.streams.video[0].average_rate
        # guard against potentially corrupted files
        if video_fps is not None:
            info["video_fps"] = float(video_fps)

    if container.streams.audio:
        audio_frames = _read_from_stream(
            container,
            audio_start_pts,
            audio_end_pts,
            pts_unit,
            container.streams.audio[0],
            {"audio": 0},
        )
        info["audio_fps"] = container.streams.audio[0].rate

    vframes_list = [frame.to_rgb().to_ndarray() for frame in video_frames]
    aframes_list = [frame.to_ndarray() for frame in audio_frames]

    if vframes_list:
        vframes = torch.as_tensor(np.stack(vframes_list))
    else:
        vframes = torch.empty((0, 1, 1, 3), dtype=torch.uint8)

    if aframes_list:
        aframes = np.concatenate(aframes_list, 1)
        aframes = torch.as_tensor(aframes)
        if pts_unit == "sec":
            audio_start_pts = int(floor(audio_start_pts * (1 / audio_timebase)))
            if audio_end_pts != float("inf"):
                audio_end_pts = int(ceil(audio_end_pts * (1 / audio_timebase)))
        aframes = _align_audio_frames(
            aframes, audio_frames, audio_start_pts, audio_end_pts
        )
    else:
        aframes = torch.empty((1, 0), dtype=torch.float32)

    if output_format == "TCHW":
        # [T,H,W,C] --> [T,C,H,W]
        vframes = vframes.permute(0, 3, 1, 2)

    return vframes, aframes, info


def scale_tensor(input_tensor, min_val, max_val):
    # Normalize tensor to 0-1
    normalized_tensor = (input_tensor - torch.min(input_tensor)) / (
        torch.max(input_tensor) - torch.min(input_tensor)
    )

    # Scale to min_val-max_val
    scaled_tensor = (max_val - min_val) * normalized_tensor + min_val

    return scaled_tensor


# FOLLOWING ARE COPIED FROM https://github.com/facebookresearch/audiocraft
# Original codebase is under MIT License


def normalize_loudness(
    wav: torch.Tensor,
    sample_rate: int,
    loudness_headroom_db: float = 14,
    loudness_compressor: bool = False,
    energy_floor: float = 2e-3,
):
    """Normalize an input signal to a user loudness in dB LKFS.
    Audio loudness is defined according to the ITU-R BS.1770-4 recommendation.

    Args:
        wav (torch.Tensor): Input multichannel audio data.
        sample_rate (int): Sample rate.
        loudness_headroom_db (float): Target loudness of the output in dB LUFS.
        loudness_compressor (bool): Uses tanh for soft clipping.
        energy_floor (float): anything below that RMS level will not be rescaled.
    Returns:
        torch.Tensor: Loudness normalized output data.
    """
    try:
        energy = wav.pow(2).mean().sqrt().item()
        if energy < energy_floor:
            return wav
        transform = torchaudio.transforms.Loudness(sample_rate)
        input_loudness_db = transform(wav).item()
        # calculate the gain needed to scale to the desired loudness level
        delta_loudness = -loudness_headroom_db - input_loudness_db
        gain = 10.0 ** (delta_loudness / 20.0)
        output = gain * wav
        if loudness_compressor:
            output = torch.tanh(output)
        assert output.isfinite().all(), (input_loudness_db, wav.pow(2).mean().sqrt())
        return output
    except Exception as e:
        print(f"Error in normalize_loudness: {e}")
        print("\twav.shape:", wav.shape)
        print(
            "\tinput_loudness_db:",
            input_loudness_db if "input_loudness_db" in locals() else None,
        )
        return wav


def _clip_wav(
    wav: torch.Tensor, log_clipping: bool = False, stem_name: Optional[str] = None
) -> None:
    """Utility function to clip the audio with logging if specified."""
    max_scale = wav.abs().max()
    if log_clipping and max_scale > 1:
        clamp_prob = (wav.abs() > 1).float().mean().item()
        print(
            f"CLIPPING {stem_name or ''} happening with proba (a bit of clipping is okay):",
            clamp_prob,
            "maximum scale: ",
            max_scale.item(),
            file=sys.stderr,
        )
    wav.clamp_(-1, 1)


def normalize_audio(
    wav: torch.Tensor,
    normalize: bool = True,
    strategy: str = "peak",
    peak_clip_headroom_db: float = 6,
    rms_headroom_db: float = 18,
    loudness_headroom_db: float = 12,
    loudness_compressor: bool = False,
    log_clipping: bool = False,
    sample_rate: Optional[int] = None,
    stem_name: Optional[str] = None,
) -> torch.Tensor:
    """Normalize the audio according to the prescribed strategy (see after).

    Args:
        wav (torch.Tensor): Audio data.
        normalize (bool): if `True` (default), normalizes according to the prescribed
            strategy (see after). If `False`, the strategy is only used in case clipping
            would happen.
        strategy (str): Can be either 'clip', 'peak', or 'rms'. Default is 'peak',
            i.e. audio is normalized by its largest value. RMS normalizes by root-mean-square
            with extra headroom to avoid clipping. 'clip' just clips.
        peak_clip_headroom_db (float): Headroom in dB when doing 'peak' or 'clip' strategy.
        rms_headroom_db (float): Headroom in dB when doing 'rms' strategy. This must be much larger
            than the `peak_clip` one to avoid further clipping.
        loudness_headroom_db (float): Target loudness for loudness normalization.
        loudness_compressor (bool): If True, uses tanh based soft clipping.
        log_clipping (bool): If True, basic logging on stderr when clipping still
            occurs despite strategy (only for 'rms').
        sample_rate (int): Sample rate for the audio data (required for loudness).
        stem_name (str, optional): Stem name for clipping logging.
    Returns:
        torch.Tensor: Normalized audio.
    """
    scale_peak = 10 ** (-peak_clip_headroom_db / 20)
    scale_rms = 10 ** (-rms_headroom_db / 20)
    if strategy == "peak":
        rescaling = scale_peak / wav.abs().max()
        if normalize or rescaling < 1:
            wav = wav * rescaling
    elif strategy == "clip":
        wav = wav.clamp(-scale_peak, scale_peak)
    elif strategy == "rms":
        mono = wav.mean(dim=0)
        rescaling = scale_rms / mono.pow(2).mean().sqrt()
        if normalize or rescaling < 1:
            wav = wav * rescaling
        _clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
    elif strategy == "loudness":
        assert sample_rate is not None, "Loudness normalization requires sample rate."
        wav = normalize_loudness(
            wav, sample_rate, loudness_headroom_db, loudness_compressor
        )
        _clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
    else:
        assert wav.abs().max() < 1
        assert (
            strategy == "" or strategy == "none"
        ), f"Unexpected strategy: '{strategy}'"
    return wav


def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to float 32 bits PCM format."""
    if wav.dtype.is_floating_point:
        return wav
    elif wav.dtype == torch.int16:
        return wav.float() / 2**15
    elif wav.dtype == torch.int32:
        return wav.float() / 2**31
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")


def _piping_to_ffmpeg(
    out_path: Union[str, Path],
    wav: torch.Tensor,
    sample_rate: int,
    flags: List[str],
):
    # ffmpeg is always installed and torchaudio is a bit unstable lately, so let's bypass it entirely.
    assert wav.dim() == 2, wav.shape
    command = (
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "f32le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(wav.shape[0]),
            "-i",
            "-",
        ]
        + flags
        + [str(out_path)]
    )
    input_ = f32_pcm(wav).t().detach().cpu().numpy().tobytes()
    sp.run(command, input=input_, check=True)


def audio_write(
    stem_name: Union[str, Path],
    wav: torch.Tensor,
    sample_rate: int,
    format: str = "wav",
    mp3_rate: int = 320,
    ogg_rate: Optional[int] = None,
    normalize: bool = True,
    strategy: str = "peak",
    peak_clip_headroom_db: float = 1,
    rms_headroom_db: float = 18,
    loudness_headroom_db: float = 14,
    loudness_compressor: bool = False,
    log_clipping: bool = True,
    make_parent_dir: bool = True,
    add_suffix: bool = True,
) -> Path:
    """Convenience function for saving audio to disk. Returns the filename the audio was written to.

    Args:
        stem_name (str or Path): Filename without extension which will be added automatically.
        wav (torch.Tensor): Audio data to save.
        sample_rate (int): Sample rate of audio data.
        format (str): Either "wav", "mp3", "ogg", or "flac".
        mp3_rate (int): kbps when using mp3s.
        ogg_rate (int): kbps when using ogg/vorbis. If not provided, let ffmpeg decide for itself.
        normalize (bool): if `True` (default), normalizes according to the prescribed
            strategy (see after). If `False`, the strategy is only used in case clipping
            would happen.
        strategy (str): Can be either 'clip', 'peak', or 'rms'. Default is 'peak',
            i.e. audio is normalized by its largest value. RMS normalizes by root-mean-square
            with extra headroom to avoid clipping. 'clip' just clips.
        peak_clip_headroom_db (float): Headroom in dB when doing 'peak' or 'clip' strategy.
        rms_headroom_db (float): Headroom in dB when doing 'rms' strategy. This must be much larger
            than the `peak_clip` one to avoid further clipping.
        loudness_headroom_db (float): Target loudness for loudness normalization.
        loudness_compressor (bool): Uses tanh for soft clipping when strategy is 'loudness'.
         when strategy is 'loudness' log_clipping (bool): If True, basic logging on stderr when clipping still
            occurs despite strategy (only for 'rms').
        make_parent_dir (bool): Make parent directory if it doesn't exist.
    Returns:
        Path: Path of the saved audio.
    """
    assert wav.dtype.is_floating_point, "wav is not floating point"
    if wav.dim() == 1:
        wav = wav[None]
    elif wav.dim() > 2:
        raise ValueError("Input wav should be at most 2 dimension.")
    assert wav.isfinite().all()
    wav = normalize_audio(
        wav,
        normalize,
        strategy,
        peak_clip_headroom_db,
        rms_headroom_db,
        loudness_headroom_db,
        loudness_compressor,
        log_clipping=log_clipping,
        sample_rate=sample_rate,
        stem_name=str(stem_name),
    )
    if format == "mp3":
        suffix = ".mp3"
        flags = ["-f", "mp3", "-c:a", "libmp3lame", "-b:a", f"{mp3_rate}k"]
    elif format == "wav":
        suffix = ".wav"
        flags = ["-f", "wav", "-c:a", "pcm_s16le"]
    elif format == "ogg":
        suffix = ".ogg"
        flags = ["-f", "ogg", "-c:a", "libvorbis"]
        if ogg_rate is not None:
            flags += ["-b:a", f"{ogg_rate}k"]
    elif format == "flac":
        suffix = ".flac"
        flags = ["-f", "flac"]
    else:
        raise RuntimeError(f"Invalid format {format}. Only wav or mp3 are supported.")
    if not add_suffix:
        suffix = ""
    path = Path(str(stem_name) + suffix)
    if make_parent_dir:
        path.parent.mkdir(exist_ok=True, parents=True)
    try:
        _piping_to_ffmpeg(path, wav, sample_rate, flags)
    except Exception:
        if path.exists():
            # we do not want to leave half written files around.
            path.unlink()
        raise
    return path
