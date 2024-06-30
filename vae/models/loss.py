import torch
import torch.nn as nn

import torch.nn.functional as F


def loss_function(x_recon, x, mu, logvar, beta: float = 1.0) -> float:
    BCE = nn.functional.binary_cross_entropy(x_recon, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return beta * KLD + BCE


def mse_loss_function(
    x_recon,
    x,
) -> float:
    MSE = F.mse_loss(x_recon, x, reduction="sum")
    return MSE
