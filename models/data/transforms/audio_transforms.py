import typing as tp
import warnings
from math import ceil

import torch
import torchaudio
import torchvision.transforms as Tv
import pyloudnorm as pyln

from utils.utils import instantiate_from_config


def get_audio_transforms(transforms_config: list) -> torch.nn.Sequential:
    """Returns a torch.nn.Sequential of audio transforms according to the config.

    Args:
        transforms_config (list): Config for the transforms.

    Returns:
        torch.nn.Sequential: Transformations to be applied to the audio.
    """
    transforms = []
    for transform_config in transforms_config:
        transform = instantiate_from_config(transform_config)
        transforms.append(transform)
    return torch.nn.Sequential(*transforms)


class AudioRandomVolume(torch.nn.Module):
    def __init__(self, p: float, **kwargs):
        super().__init__()
        transform = torchaudio.transforms.Vol(**kwargs)
        self.transform = Tv.RandomApply([transform], p)

    def forward(self, wav):
        wav = self.transform(wav)
        return wav


class AudioRandomLowpassFilter(torch.nn.Module):
    def __init__(self, p: float, cutoff_freq: float, Q: float = 0.707, sr=24000):
        super().__init__()
        self.p = p
        self.cutoff_freq = cutoff_freq
        self.Q = Q
        self.sr = sr

    def forward(self, wav):
        if self.p > torch.rand(1):
            wave = wav.unsqueeze(0)
            wave = torchaudio.functional.lowpass_biquad(
                wave, self.sr, self.cutoff_freq, self.Q
            )
            wav = wave.squeeze(0)
        return wav


class AudioRandomPitchShift(torch.nn.Module):
    def __init__(self, p: float, shift: int, sr: int) -> None:
        super().__init__()
        self.p = p
        self.shift = shift
        self.sr = sr

    def forward(self, wav):
        if self.p > torch.rand(1):
            effects = [["pitch", f"{self.shift}"], ["rate", f"{self.sr}"]]
            wave = wav.unsqueeze(0)
            wave, _ = torchaudio.sox_effects.apply_effects_tensor(
                wave, self.sr, effects
            )
            wav = wave.squeeze(0)
        return wav


class AudioRandomReverb(torch.nn.Module):
    def __init__(self, p: float, sr: int = 24000) -> None:
        super().__init__()
        self.p = p
        self.sr = sr
        self.effects = [["reverb", "-w"]]

    def forward(self, wav):
        if self.p > torch.rand(1):
            wave = wav.unsqueeze(0)
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                wave, self.sr, self.effects
            )
            wav = wav.mean(dim=0)
        return wav


class AudioRandomGaussNoise(torch.nn.Module):
    def __init__(self, p: float, amplitude=0.01) -> None:
        super().__init__()
        self.p = p
        self.amplitude = amplitude

    def forward(self, wav):
        if self.p > torch.rand(1):
            wave = wav
            noise = torch.randn_like(wave, dtype=wave.dtype)
            wav = wave + self.amplitude * noise
        return wav


class AudioStandardNormalize(torch.nn.Module):
    def __init__(
        self,
        mean: tp.Optional[float] = 0,
        std: tp.Optional[float] = 1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.mean = mean
        self.std = std

    def forward(self, wav):
        wav = (wav - self.mean) / (self.std + self.eps)
        return wav


class AudioLoudnessNormalize(torch.nn.Module):
    def __init__(self, target_loudness: float = -24.0, sr: int = 24000) -> None:
        super().__init__()
        self.target_loudness = target_loudness
        self.a_sr = sr

    def forward(self, wav: torch.Tensor):
        wav = wav.numpy()
        meter = pyln.Meter(self.a_sr)
        loudness = meter.integrated_loudness(wav)
        with warnings.catch_warnings():  # throws 'clipped samples' warning
            warnings.simplefilter("ignore")
            wav = pyln.normalize.loudness(wav, loudness, self.target_loudness)
        wav = torch.from_numpy(wav)
        return wav


class AudioRandomPhaser(torch.nn.Module):
    def __init__(self, p: float, sr: int) -> None:
        super().__init__()
        self.p = p
        self.sr = sr

    def forward(self, wav: torch.Tensor):
        if self.p > torch.rand(1):
            wav = torchaudio.functional.phaser(wav, self.sr)
        return wav


class AudioUnsqueeze(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, wav: torch.Tensor):
        return wav.unsqueeze(self.dim)


class AudioStereoToMono(torch.nn.Module):
    def __init__(self, keepdim=True) -> None:
        super().__init__()
        self.keepdim = keepdim

    def forward(self, wav: torch.Tensor):
        return wav.mean(dim=0, keepdim=self.keepdim)


class AudioResample(torch.nn.Module):
    def __init__(self, target_sr: int, clip_duration: float) -> None:
        super().__init__()
        self.target_sr = target_sr
        self.clip_duration = clip_duration

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        original_sr = int(wav.shape[-1] / self.clip_duration)
        wav = torchaudio.transforms.Resample(
            orig_freq=original_sr, new_freq=self.target_sr
        )(wav)
        return wav


class AudioTrim(torch.nn.Module):
    def __init__(self, duration: float, sr: int) -> None:
        super().__init__()
        self.duration = duration
        self.sr = sr

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return wav[..., : ceil(self.duration * self.sr)]
