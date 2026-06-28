import torch
import torch.nn as nn

import torchaudio

from .modules.audio_utils import get_mel_transform
from .model import EncoderDecoder, remove_all_weight_norms

class VAEInferenceWrapper(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = EncoderDecoder()
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

        self.transform = self.get_transform()
        self.sample_rate = self.model.config.sample_rate

        remove_all_weight_norms(self.model)

    @staticmethod
    def get_transform():
        return get_mel_transform()[0]

    def encode(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0) # [1, t]
        x = self.transform(x) # [b, n_mels, t]
        return self.model.encode(x)[0]
    
    def decode(self, x):
        return self.model.decode(x)[0].squeeze(1) # [b, t]
    
    @torch.inference_mode()
    def reconstruct(self, audio, sample_rate):
        device = next(self.model.parameters()).device

        if sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, self.sample_rate)

        latent = self.encode(audio.to(device))
        reconstructed = self.decode(latent)
        return reconstructed, (latent,)