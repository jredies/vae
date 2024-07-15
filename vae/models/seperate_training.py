import typing
import logging

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from vae.models.training import (
    select_device,
    estimate_log_marginal,
    get_view,
    standard_loss,
    reconstruction_loss,
    initialize_scheduler,
    set_loss,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def training_step(
    input_dim: int,
    device: str,
    vae: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loss: float,
    train_recon: float,
    train_selbo: float,
    data: typing.Any,
    latent_noise: float = 0.0,
    clip_gradient: bool = False,
    norm_gradient: bool = False,
    beta: float = 1.0,
    loss_fn: typing.Callable = standard_loss,
    cnn=False,
) -> typing.Tuple[float, float]:

    data = get_view(device=device, input_dim=input_dim, x=data, cnn=cnn)

    optimizer.zero_grad()
    x_recon, mu, logvar = vae.forward(x=data, noise_parameter=latent_noise)

    loss = loss_fn(
        x_recon=x_recon,
        x=data,
        mu=mu,
        logvar=logvar,
        beta=beta,
        model=vae,
        cnn=cnn,
    ).to(device)
    recon = reconstruction_loss(x_recon=x_recon, x=data).to(device)
    standard_elbo = standard_loss(
        x_recon=x_recon, x=data, mu=mu, logvar=logvar, cnn=cnn
    ).to(device)
    loss.backward()

    train_loss += loss.item()
    train_recon += recon.item()
    train_selbo += standard_elbo.item()

    if clip_gradient:
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)

    if norm_gradient:
        epsilon = 1e-8
        _ = [
            param.grad.div_(torch.norm(param.grad) + epsilon)
            for param in vae.parameters()
            if param.grad is not None
        ]

    optimizer.step()

    return train_loss, train_recon, train_selbo


def train_vae(
    model_path: str,
    file_name: str,
    vae: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    dim: np.ndarray,
    base_learning_rate: float = 1e-3,
    patience: int = 10,
    epochs: int = 300,
    scheduler_type: str = "plateau",
    plateau_patience: int = 5,
    step_size: int = 10,
    gamma: float = 0.1,
    gaussian_noise: float = 0.0,
    salt_and_pepper_noise: float = 0.0,
    latent_noise: float = 0.0,
    clip_gradient: bool = False,
    norm_gradient: bool = False,
    loss_type: str = "standard",
    iw_samples: int = 5,
    cnn: bool = False,
    annealing_start: int = 0,
    annealing_end: int = 0,
    annealing_type: str = "linear",
):
    device = select_device()
    vae = vae.to(device)
    input_dim = np.prod(dim)

    writer = SummaryWriter(model_path)
    df_stats = pd.DataFrame()

    optimizer_enc = optim.Adam(
        vae.encoder.parameters(),
        lr=base_learning_rate,
        eps=1e-4,
    )

    optimizer_dec = optim.Adam(
        vae.decoder.parameters(),
        lr=base_learning_rate,
        eps=1e-4,
    )
    scheduler = initialize_scheduler(
        scheduler_type=scheduler_type,
        plateau_patience=plateau_patience,
        step_size=step_size,
        gamma=gamma,
        optimizer=optimizer_enc,
    )
    scheduler = initialize_scheduler(
        scheduler_type=scheduler_type,
        plateau_patience=plateau_patience,
        step_size=step_size,
        gamma=gamma,
        optimizer=optimizer_dec,
    )

    loss_fn = set_loss(loss_type, iw_samples, cnn)
