import typing as tp
from pathlib import Path

import dac
import torch
from torch.nn import Module


MODEL_SR = [16000, 24000, 44000, 44100]


class DacModelWrapper(Module):
    def __init__(
        self, model_sr: int = 24000, ckpt_path: tp.Optional[tp.Union[str]] = None
    ) -> None:
        super().__init__()
        assert model_sr in MODEL_SR, "Invalid model samplerate"
        self.model_sr = model_sr

        if ckpt_path is not None and Path(ckpt_path).exists():
            model_path = ckpt_path
        else:
            model_path = dac.utils.download(model_type=f"{model_sr // 1000}khz")

        self.model = dac.DAC.load(model_path)

    def forward(self, wav: torch.Tensor):
        return self.encode(wav)

    @torch.no_grad()
    def encode(self, wav: torch.Tensor):
        if wav.ndim < 2:
            wav = wav.unsqueeze(0)
        if wav.ndim < 3:
            wav = wav.unsqueeze(0)
        wav = self.model.preprocess(wav, self.model_sr)
        with torch.no_grad():
            _, codes, _, _, _ = self.model.encode(wav)
        return codes

    @torch.no_grad()
    def decode(self, codes: tp.Union[torch.Tensor, tp.List[tp.Tuple[torch.Tensor]]]):
        if type(codes) == list:  # for compatibility with EnCodec
            codes = codes[0][0]
        with torch.no_grad():
            z, _, _ = self.model.quantizer.from_codes(codes)
            audio = self.model.decode(z)
        return audio

    @property
    def sample_rate(self):
        return self.model.sample_rate

    @property
    def channels(self):
        return 1

    @property
    def frame_rate(self):
        return None  # TODO: figure this out


if __name__ == "__main__":
    import torchaudio

    # Init model
    m = DacModelWrapper(model_sr=24000)
    m.to("cuda:1")
    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load("./data/test/test_24k.wav")
    wav = wav.unsqueeze(0)
    wav = wav.to("cuda:1")
    # Encode
    codes = m.encode(wav)
    # Decode
    audio = m.decode(codes)
    print(audio.shape)
    # Save the audio
    audio = audio.squeeze(0).cpu()
    torchaudio.save("./data/test/test_24k_reconstructed.wav", audio, sr)
