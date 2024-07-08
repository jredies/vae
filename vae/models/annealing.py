import itertools
import logging
import typing

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from vae.models.training import (
    calculate_test_loss,
    calculate_stats,
    get_loaders,
    select_device,
    initialize_scheduler,
    training_step,
    update_scheduler,
    log_training_epoch,
    write_all_stats,
)
from vae.models.simple_vae import create_vae_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_vae(
    model_path: str,
    vae: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    dim: np.ndarray,
    base_learning_rate: float = 1e-3,
    epochs: int = 200,
    scheduler_type: str = "plateau",
    plateau_patience: int = 5,
    step_size: int = 10,
    gamma: float = 1.0,
    gaussian_noise: float = 0.0,
    salt_and_pepper_noise: float = 0.0005,
    clip_gradient: bool = False,
    norm_gradient: bool = False,
    annealing_start: int = 0,
    annealing_stop: int = 100,
    annealing_method: str = "linear",
    patience: int = 10,
):
    device = select_device()
    vae = vae.to(device)
    input_dim = np.prod(dim)

    writer = SummaryWriter(model_path)
    df_stats = pd.DataFrame()

    optimizer = optim.Adam(vae.parameters(), lr=base_learning_rate)

    scheduler = initialize_scheduler(
        scheduler_type=scheduler_type,
        plateau_patience=plateau_patience,
        step_size=step_size,
        gamma=gamma,
        optimizer=optimizer,
    )

    best_val_loss = float("inf")
    patience_counter = 0

    vae.train()
    for epoch in range(epochs):
        train_loss, train_mse = 0.0, 0.0

        beta = calculate_beta(
            annealing_start=annealing_start,
            annealing_stop=annealing_stop,
            annealing_method=annealing_method,
            epoch=epoch,
        )

        log.info(f"Epoch: {epoch}, Beta: {beta}")
        if epoch == annealing_stop:
            set_learning_rate(optimizer, base_learning_rate)
            scheduler = initialize_scheduler(
                optimizer=optimizer,
                scheduler_type=scheduler_type,
                plateau_patience=plateau_patience,
                step_size=step_size,
                gamma=gamma,
            )

        for _, (data, _) in enumerate(train_loader):
            train_loss, train_mse = training_step(
                input_dim=input_dim,
                device=device,
                vae=vae,
                optimizer=optimizer,
                train_loss=train_loss,
                train_bce=train_mse,
                data=data,
                gaussian_noise=gaussian_noise,
                salt_and_pepper_noise=salt_and_pepper_noise,
                norm_gradient=norm_gradient,
                clip_gradient=clip_gradient,
                beta=beta,
            )

        n_train = len(train_loader.dataset)
        train_loss /= n_train
        train_mse /= n_train

        vae.eval()
        val_loss, val_mse = calculate_stats(
            vae=vae,
            loader=validation_loader,
            device=device,
            input_dim=input_dim,
            beta=beta,
        )
        test_loss, test_mse = calculate_test_loss(
            vae=vae,
            test_loader=test_loader,
            device=device,
            input_dim=input_dim,
        )

        update_scheduler(
            scheduler_type=scheduler_type,
            gamma=gamma,
            scheduler=scheduler,
            val_loss=val_loss,
        )

        write_all_stats(
            writer=writer,
            df_stats=df_stats,
            epoch=epoch,
            train_loss=train_loss,
            train_bce=train_mse,
            val_loss=val_loss,
            val_kld=val_mse,
            test_loss=test_loss,
            test_bce=test_mse,
        )
        log_training_epoch(
            optimizer=optimizer,
            best_val_loss=best_val_loss,
            epoch=epoch,
            train_loss=train_loss,
            train_bce=train_mse,
            val_loss=val_loss,
            val_mse=val_mse,
        )

        if epoch > annealing_stop:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                log.info("Early stopping triggered.")
                break

    writer.close()
    return df_stats


def calculate_beta(
    annealing_start: int,
    annealing_stop: int,
    annealing_method: str,
    epoch: int,
) -> float:
    beta = 1.0

    if annealing_method == "linear":
        if (epoch >= annealing_start) and (epoch <= annealing_stop):
            rel_position = epoch - annealing_start
            length = annealing_stop - annealing_start
            if length == 0:
                beta = 1.0
            else:
                beta = float(rel_position) / float(length)
    elif annealing_method == "step":
        if (epoch >= annealing_start) and (epoch <= annealing_stop):
            beta = 0.0
    else:
        raise ValueError(f"Unknown annealing method: {annealing_method}")
    return beta


def main():
    path = "logs/aug"
    train_loader, validation_loader, test_loader, dim = get_loaders()

    latent_factor = 0.05

    start = [
        0,
        # 10,
        # 20,
    ]
    lengths = [
        0,
        # 50,
        # 75,
        # 100,
    ]
    params = itertools.product(start, lengths)
    params = list(params)
    params = params

    for start, length in params:
        vae = create_vae_model(dim=dim, latent_dim_factor=latent_factor)

        df_stats = train_vae(
            vae=vae,
            train_loader=train_loader,
            validation_loader=validation_loader,
            test_loader=test_loader,
            dim=dim,
            model_path=path,
            gamma=0.25,
            annealing_start=start,
            annealing_stop=start + length,
            annealing_method="step",
        )

        df_stats.to_csv(f"{path}/step_start-{start}_len-{length}.csv")


if __name__ == "__main__":
    main()
