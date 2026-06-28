import math

import torch

from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrize import remove_parametrizations, is_parametrized

from .modules.stable_audio import OobleckEncoder, OobleckDecoder
from .modules.audio_utils import get_mel_transform

from typing import List
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    decoder_hidden_channels: int = 128
    decoder_c_mults: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    decoder_strides: List[int] = field(default_factory=lambda: [2, 4, 6, 10, 10])

    encoder_hidden_channels: int = 256
    encoder_c_mults: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    encoder_strides: List[int] = field(default_factory=lambda: [2, 2, 2, 2])

    bottleneck_channels: int = 128
    
    def __post_init__(self):
        _, self.mel_config = get_mel_transform()
        self.input_channels = self.mel_config.n_mels
        self.sample_rate = self.mel_config.sample_rate

        self.input_downsample_factor = math.prod(self.encoder_strides)
        self.output_upsample_factor = math.prod(self.decoder_strides)
        self.frame_rate = self.sample_rate / self.output_upsample_factor

class EncoderDecoder(nn.Module):
    def __init__(self, config: ModelConfig = None):
        super().__init__()
        self.config = config or ModelConfig()

        self.encoder = OobleckEncoder(
            in_channels=self.config.input_channels,
            channels=self.config.encoder_hidden_channels,
            c_mults=self.config.encoder_c_mults,
            strides=self.config.encoder_strides,
        )

        self.bottleneck = nn.Conv1d(
            in_channels=self.config.encoder_hidden_channels * self.config.encoder_c_mults[-1],
            out_channels=self.config.bottleneck_channels,
            kernel_size=1
        )
        self.norm =  nn.LayerNorm(self.config.bottleneck_channels, elementwise_affine=False)

        self.decoder = OobleckDecoder(
            channels=self.config.decoder_hidden_channels,
            latent_dim=self.config.bottleneck_channels,
            c_mults=self.config.decoder_c_mults,
            strides=self.config.decoder_strides,
        )

    def encode(self, input_values):
        encoder_output, encoder_hidden_states = self.encoder(input_values)

        encoder_output = self.bottleneck(encoder_output)
        encoder_output = self.norm(encoder_output.transpose(1, 2)).transpose(1, 2)

        return encoder_output, (encoder_hidden_states,)
    
    def decode(self, input_values):
        decoder_output = self.decoder(input_values)
        return decoder_output, (decoder_output,)
    
    def inference(self, input_values):
        encoder_output, _ = self.encode(input_values)
        decoder_output, _ = self.decode(encoder_output)
        return decoder_output
    
def remove_all_weight_norms(module: nn.Module, verbose=False):
    for name, submodule in module.named_children():
        remove_all_weight_norms(submodule)
        
        if is_parametrized(submodule, 'weight'):
            remove_parametrizations(submodule, 'weight')
            if verbose:
                print(f"Removed weight norm from {name}")
    return module
        
if __name__ == '__main__':
    model = EncoderDecoder()
