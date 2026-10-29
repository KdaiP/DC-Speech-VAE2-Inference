import math

import torch
import torch.nn as nn

from torch.nn.utils.parametrizations import weight_norm

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

def get_activation() -> nn.Module:
    return nn.SiLU(inplace=True)

class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), device=x.device, dtype=x.dtype)
        return x + self.weight * noise

class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        
        self.dilation = dilation

        padding = (dilation * (7-1)) // 2

        self.layers = nn.Sequential(
            get_activation(),
            WNConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, dilation=dilation, padding=padding),
            get_activation(),
            WNConv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
        )

    def forward(self, x):
        return x + self.layers(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=1),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=3),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=9),
            get_activation(),
            WNConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2)),
        )

    def forward(self, x):
        return self.layers(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.up = nn.Sequential(
            get_activation(),
            WNConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                        dilation=1),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                        dilation=3),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                        dilation=9),
        )

        self.noise1 = ApplyNoise(out_channels)
        self.noise2 = ApplyNoise(out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.noise1(x)
        x = self.layers(x)
        x = self.noise2(x)
        return x

class OobleckDecoder(nn.Module):
    def __init__(self, 
                 out_channels=1, 
                 channels=128, 
                 latent_dim=32, 
                 c_mults = [1, 2, 4, 8], 
                 strides = [2, 4, 8, 8],
        ):
        super().__init__()
        self.out_channels = out_channels

        c_mults = [1] + c_mults
        
        self.depth = len(c_mults)

        layers = [
            WNConv1d(in_channels=latent_dim, out_channels=c_mults[-1]*channels, kernel_size=7, padding=3),
        ]
        
        for i in range(self.depth-1, 0, -1):
            layers += [DecoderBlock(
                in_channels=c_mults[i]*channels, 
                out_channels=c_mults[i-1]*channels, 
                stride=strides[i-1], 
                )
            ]

        layers += [
            get_activation(),
            WNConv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7, padding=3, bias=False)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckEncoder(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 channels=128, 
                 c_mults = [1, 2, 4, 8], 
                 strides = [2, 4, 8, 8],
        ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels=c_mults[-1] * channels
          
        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        self.pre = WNConv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7, padding=3)
        
        blocks = []
        for i in range(self.depth-1):
            blocks += [EncoderBlock(in_channels=c_mults[i]*channels, out_channels=c_mults[i+1]*channels, stride=strides[i])]

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.pre(x)

        hidden_states = []
        for block_idx, block in enumerate(self.blocks):
            hidden_states.append(x)
            x = block(x)
        return x, hidden_states
    