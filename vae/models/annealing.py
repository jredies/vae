import pathlib
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
    calculate_stats,
    get_loaders,
    select_device,
    initialize_scheduler,
    training_step,
    update_scheduler,
    log_training_epoch,
    write_all_stats,
    estimate_log_marginal,
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
    file_name: str,
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
        train_loss, train_recon, train_selbo, mini_batches = 0.0, 0.0, 0.0, 0

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
            train_loss, train_recon, train_selbo = training_step(
                input_dim=input_dim,
                device=device,
                vae=vae,
                optimizer=optimizer,
                train_loss=train_loss,
                train_recon=train_recon,
                train_selbo=train_selbo,
                data=data,
                gaussian_noise=gaussian_noise,
                salt_and_pepper_noise=salt_and_pepper_noise,
                norm_gradient=norm_gradient,
                clip_gradient=clip_gradient,
                beta=beta,
            )
            mini_batches += 1

        train_loss /= mini_batches
        train_recon /= mini_batches
        train_selbo /= mini_batches

        vae.eval()
        val_loss, val_recon, val_selbo = calculate_stats(
            vae=vae,
            loader=validation_loader,
            device=device,
            input_dim=input_dim,
        )
        test_loss, test_recon, test_selbo = calculate_stats(
            vae=vae,
            loader=test_loader,
            device=device,
            input_dim=input_dim,
        )

        update_scheduler(
            scheduler_type=scheduler_type,
            gamma=gamma,
            scheduler=scheduler,
            val_loss=val_loss,
            epoch=epoch,
            annealing_start=annealing_start,
            annealing_end=annealing_stop,
            optimizer=optimizer,
        )

        log_training_epoch(
            optimizer=optimizer,
            best_val_loss=best_val_loss,
            best_val_selbo=best_val_loss,
            epoch=epoch,
            train_loss=train_loss,
            train_recon=train_recon,
            train_selbo=train_selbo,
            val_loss=val_loss,
            val_recon=val_recon,
            val_selbo=val_selbo,
            vae=vae,
            beta=beta,
        )

        early_stopping = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        elif epoch < annealing_stop:
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch > annealing_stop:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                log.info("Early stopping triggered.")
                break

        epoch_mod = epoch % 50 == 0

        if early_stopping or epoch_mod:
            lm_val = estimate_log_marginal(
                model=vae,
                data_loader=validation_loader,
                device=device,
                input_dim=input_dim,
            )
            lm_train = estimate_log_marginal(
                model=vae,
                data_loader=train_loader,
                device=device,
                input_dim=input_dim,
            )
            lm_test = estimate_log_marginal(
                model=vae,
                data_loader=test_loader,
                device=device,
                input_dim=input_dim,
            )
            log.info(
                f"Log marginal likelihood: Val {lm_val:.4f} Tr {lm_train:.4f} Test {lm_test:.4f}"
            )

        write_all_stats(
            writer=writer,
            df_stats=df_stats,
            epoch=epoch,
            train_loss=train_loss,
            train_lm=lm_train,
            train_recon=train_recon,
            train_selbo=train_selbo,
            val_loss=val_loss,
            val_lm=lm_val,
            val_recon=val_recon,
            val_selbo=val_selbo,
            test_loss=test_loss,
            test_lm=lm_test,
            test_recon=test_recon,
            test_selbo=test_selbo,
            best_val_loss=best_val_loss,
            beta=beta,
        )
        if (epoch % 20 == 0) or early_stopping:
            df_stats.to_csv(pathlib.Path(model_path) / f"{file_name}.csv")

        if early_stopping:
            break

    writer.close()


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
        5,
        50,
        75,
        100,
        0,
    ]
    params = itertools.product(start, lengths)
    params = list(params)
    params = params

    for start, length in params:
        vae = create_vae_model(dim=dim, latent_dim_factor=latent_factor)
        filename = f"linear_start-{start}_len-{length}"

        df_stats = train_vae(
            vae=vae,
            file_name=filename,
            train_loader=train_loader,
            validation_loader=validation_loader,
            test_loader=test_loader,
            dim=dim,
            model_path=path,
            gamma=0.25,
            annealing_start=start,
            annealing_stop=start + length,
            annealing_method="linear",
            patience=15,
            plateau_patience=7,
            epochs=400,
            scheduler_type="plateau",
        )


if __name__ == "__main__":
    main()
