<p align="center">
  <img src="./assets/banner.jpg" alt="DC-Speech-VAE2 Banner" width="100%">
</p>

# DC-Speech-VAE2-Inference

[![code](https://img.shields.io/badge/Github-Code-keygen.svg?logo=github)](https://github.com/KdaiP/DC-Speech-VAE2-Inference)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging_Face-Model-blue.svg)](https://huggingface.co/KdaiP/DC-Speech-VAE2)

This repository provides the inference code for **DC-Speech-VAE2**, an upgraded version of [DC-Speech-VAE](https://github.com/KdaiP/DC-Speech-VAE).
DC-Speech-VAE2 compresses **48 kHz** audio into a **10 Hz** **128 channels** latent representation.

By applying aggressive latent augmentation during training, DC-Speech-VAE2 learns a latent representation that is easier to model for downstream diffusion/flow-matching based generative tasks. Compared with DC-Speech-VAE, it achieves approximately a **30% reduction in downstream flow-matching loss**.

## Overview
<p align="center">
  <img src="assets/model_structure.jpg" width="700">
  <br>
  <em></em>
</p>

DC-Speech-VAE2 adopts the architecture of [DAC](https://github.com/descriptinc/descript-audio-codec) for end-to-end training and inference. The model takes high-quality mel spectrogram as input and directly reconstructs the waveform.

Unlike [DC-Speech-VAE](https://github.com/KdaiP/DC-Speech-VAE), DC-Speech-VAE2 does not require an additional mel vocoder such as [BigVGAN](https://github.com/NVIDIA/BigVGAN).

## Evaluation results

We compare DC-Speech-VAE2 with [BigVGAN V2](https://github.com/NVIDIA/BigVGAN) and [SAME-L](https://huggingface.co/stabilityai/SAME-L), the [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3) VAE.

Inference results for all three models are available here: [🤗 Download examples 🤗](https://huggingface.co/KdaiP/DC-Speech-VAE2/blob/main/subjective_eval_audio.zip).

We recommend listening to the audio samples to compare the perceptual quality of each model.

| Model | Sample Rate | Frame Rate | Channels | Encoder Params | Decoder Params | PESQ ↑ | MSTFT ↓ | Encoder RTF | Decoder RTF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BigVGAN V2 | 44.1 kHz | 86.133 Hz | **128** | N/A | 122.185M | **4.0134** | **0.9440** | N/A | 0.14996 |
| SAME-L | 44.1 kHz | 10.767 Hz | 256 | 426.064M | 426.063M | <u>2.8292</u> | 1.1213 | 0.10363 | 0.10191 |
| **DC-Speech-VAE2** | **48 kHz** | **10.000 Hz** | **128** | **46.529M** | **90.021M** | 2.3719 | <u>1.1029</u> | **0.00214** | **0.03842** |

RTF is measured on an RTX 5060.

<p align="center">
  <img src="assets/spec_compare.png" width="700">
  <br>
  <em></em>
</p>

BigVGAN V2 and SAME-L show periodic high-frequency artifacts; DC-Speech-VAE2 is visually cleaner.

## Quick start

### 1. Download pretrained model

Download the [🤗 pretrained checkpoint 🤗](https://huggingface.co/KdaiP/DC-Speech-VAE2/blob/main/checkpoint.pt) and place it at:

```bash
./pretrained/checkpoint.pt
```

### 2. Install dependencies

The only required dependencies are **PyTorch** and **torchaudio**.

Please follow the official PyTorch installation guide for your platform and hardware:

<https://pytorch.org/get-started/locally>

### 3. Inference

``` python
import torch
import torchaudio

from dc_speech_vae.wrapper import VAEInferenceWrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAEInferenceWrapper('./pretrained/checkpoint.pt').to(device)

audio_input_path = 'test.wav'
audio_output_path = 'reconstructed.wav'

audio, sample_rate = torchaudio.load(audio_input_path)

reconstructed_audio = model.reconstruct(audio, sample_rate)[0]
torchaudio.save(audio_output_path, reconstructed_audio.cpu(), model.sample_rate)
print(f"Reconstructed audio saved to {audio_output_path} at sample rate {model.sample_rate}.")
```

## Reference

This project builds on ideas and components from the following works and repositories:

- [HiFi-GAN](https://github.com/jik876/hifi-gan)
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
- [DAC](https://github.com/descriptinc/descript-audio-codec)
- [HiFi-VAE-GAN](https://github.com/OpenNSP/Hifi-vaegan)
- [RefineGAN](https://arxiv.org/abs/2111.00962)

## Spetial Thanks

- [zhiyongchenGREAT](https://github.com/zhiyongchenGREAT)
- [aizhiqi-work](https://github.com/aizhiqi-work)
- [ylzz1997](https://github.com/ylzz1997)
- [CNChTu](https://github.com/CNChTu)