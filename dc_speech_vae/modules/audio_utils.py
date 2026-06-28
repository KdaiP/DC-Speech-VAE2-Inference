import torch
from torch import Tensor
import torch.nn as nn
import torchaudio

from dataclasses import dataclass, asdict

def get_mel_transform():
    mel_config = MelConfig()
    return LogMelSpectrogram(**asdict(mel_config)), mel_config
            
@dataclass
class MelConfig:
    sample_rate: int = 48000
    n_fft: int = 1200
    win_length: int = 1200
    hop_length: int = 300
    f_min: float = 40.0
    f_max: float = None
    pad: int = 0
    n_mels: int = 200
    center: bool = False
    pad_mode: str = "reflect"
    mel_scale: str = "slaney"
    
    def __post_init__(self):
        if self.pad == 0:
            self.pad = (self.n_fft - self.hop_length) // 2
            
def compress(x: Tensor) -> Tensor:
    return torch.log(torch.clamp(x, min=1e-5))
            
class SpecExtractor(nn.Module):
    def __init__(self, mel_config: MelConfig):
        super().__init__()
        n_fft=mel_config.n_fft
        win_length=mel_config.win_length
        hop_length=mel_config.hop_length
        pad=mel_config.pad
        center=mel_config.center
        pad_mode=mel_config.pad_mode
        
        self.transform = LinearSpectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, pad=pad, center=center, pad_mode=pad_mode)
        
    def forward(self, x: Tensor, return_energy=False) -> Tensor:
        spec = self.transform(x)
        energy = compress(spec.sum(dim=1)) if return_energy else None
        spec = compress(spec)

        return spec, energy # shape: [batch_size, time]

class LinearSpectrogram(nn.Module):
    def __init__(self, n_fft, win_length, hop_length, pad, center, pad_mode):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.pad = pad
        self.center = center
        self.pad_mode = pad_mode
        
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, waveform: Tensor) -> Tensor:
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)
        waveform = torch.nn.functional.pad(waveform.unsqueeze(1), (self.pad, self.pad), self.pad_mode).squeeze(1)
        spec = torch.stft(waveform, self.n_fft, self.hop_length, self.win_length, self.window, self.center, self.pad_mode, False, True, True)
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
        return spec

class LogMelSpectrogram(nn.Module):
    def __init__(self, sample_rate, n_fft, win_length, hop_length, f_min, f_max, pad, n_mels, center, pad_mode, mel_scale, **kwargs):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.pad = pad
        self.n_mels = n_mels
        self.center = center
        self.pad_mode = pad_mode
        self.mel_scale = mel_scale
        
        self.spectrogram = LinearSpectrogram(n_fft, win_length, hop_length, pad, center, pad_mode)
        self.mel_scale = torchaudio.transforms.MelScale(n_mels, sample_rate, f_min, f_max, (n_fft//2)+1, mel_scale, mel_scale)

    def compress(self, x: Tensor) -> Tensor:
        return torch.log(torch.clamp(x, min=1e-5))

    def decompress(self, x: Tensor) -> Tensor:
        return torch.exp(x)

    def forward(self, x: Tensor) -> Tensor:
        linear_spec = self.spectrogram(x)
        x = self.mel_scale(linear_spec)
        x = self.compress(x)
        return x