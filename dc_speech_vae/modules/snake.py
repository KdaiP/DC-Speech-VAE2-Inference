import torch
import torch.nn as nn

def snake_beta(x, alpha, beta):
    return x + (1.0 / (beta + 0.000000001)) * torch.pow(torch.sin(x * alpha), 2)

class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super().__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale: # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(1, in_features, 1) * alpha)
            self.beta = nn.Parameter(torch.zeros(1, in_features, 1) * alpha)
        else: # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(1, in_features, 1) * alpha)
            self.beta = nn.Parameter(torch.ones(1, in_features, 1) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha
        beta = self.beta
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = snake_beta(x, alpha, beta)

        return x