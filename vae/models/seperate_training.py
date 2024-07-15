import pathlib
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
    initialize_scheduler,
    calculate_stats,
    log_training_epoch,
    get_loaders,
    create_vae_model,
    write_all_stats,
    update_scheduler,
)
from vae.models.loss import iwae_loss_fast, standard_loss

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
    optimizer_dec: torch.optim.Optimizer,
    optimizer_enc: torch.optim.Optimizer,
    data: typing.Any,
    latent_noise: float = 0.0,
    cnn=False,
    iw_samples: int = 5,
    beta=0.1,
) -> typing.Tuple[float, float]:
    data = get_view(device=device, input_dim=input_dim, x=data, cnn=cnn)
    recon_x, mu, logvar = vae.forward(x=data, noise_parameter=latent_noise)
    vae_loss_enc = standard_loss(x_recon=recon_x, x=data, mu=mu, logvar=logvar, cnn=cnn)
    iwae_loss = iwae_loss_fast(model=vae, x=data, num_samples=iw_samples)

    # Combined PIWAE Loss for the encoder
    piwae_loss = beta * vae_loss_enc + (1 - beta) * iwae_loss
    optimizer_enc.zero_grad()
    piwae_loss.backward()
    optimizer_enc.step()

    # VAE Loss for the decoder
    vae_loss_dec = standard_loss(x_recon=recon_x, x=data, mu=mu, logvar=logvar, cnn=cnn)
    optimizer_dec.zero_grad()
    vae_loss_dec.backward()
    optimizer_dec.step()

    return piwae_loss.item(), vae_loss_dec.item()


def training_step(
    input_dim: int,
    device: str,
    vae: nn.Module,
    optimizer_dec: torch.optim.Optimizer,
    optimizer_enc: torch.optim.Optimizer,
    data: typing.Any,
    latent_noise: float = 0.0,
    cnn=False,
    iw_samples: int = 5,
) -> typing.Tuple[float, float]:
    data = get_view(device=device, input_dim=input_dim, x=data, cnn=cnn)

    # Forward pass
    recon_x, mu, logvar = vae.forward(x=data, noise_parameter=latent_noise)

    # IWAE Loss for the encoder
    iwae_loss = iwae_loss_fast(model=vae, x=data, num_samples=iw_samples)
    optimizer_enc.zero_grad()
    iwae_loss.backward()
    optimizer_enc.step()

    # VAE Loss for the decoder
    vae_loss = standard_loss(x_recon=recon_x, x=data, mu=mu, logvar=logvar, cnn=cnn)
    optimizer_dec.zero_grad()
    vae_loss.backward(retain_graph=True)
    optimizer_dec.step()


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
    iw_samples: int = 5,
    cnn: bool = False,
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
    scheduler_enc = initialize_scheduler(
        scheduler_type=scheduler_type,
        plateau_patience=plateau_patience,
        step_size=step_size,
        gamma=gamma,
        optimizer=optimizer_enc,
    )
    scheduler_dec = initialize_scheduler(
        scheduler_type=scheduler_type,
        plateau_patience=plateau_patience,
        step_size=step_size,
        gamma=gamma,
        optimizer=optimizer_dec,
    )

    best_val_loss = float("inf")
    best_val_selbo = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        vae.train()
        for _, (data, _) in enumerate(train_loader):
            training_step(
                input_dim=input_dim,
                device=device,
                vae=vae,
                optimizer_dec=optimizer_dec,
                optimizer_enc=optimizer_enc,
                data=data,
                cnn=cnn,
                iw_samples=iw_samples,
            )

        vae.eval()
        kwargs = {
            "vae": vae,
            "device": device,
            "input_dim": input_dim,
            "loss_fn": iwae_loss_fast,
            "cnn": cnn,
        }

        train_loss, train_recon, train_selbo = calculate_stats(
            loader=train_loader, **kwargs
        )
        val_loss, val_recon, val_selbo = calculate_stats(
            loader=validation_loader, **kwargs
        )
        test_loss, test_recon, test_selbo = calculate_stats(
            loader=test_loader, **kwargs
        )

        kwargs = {
            "scheduler_type": scheduler_type,
            "gamma": gamma,
            "val_loss": val_loss,
            "epoch": epoch,
            "annealing_end": 0,
            "annealing_start": 0,
        }

        update_scheduler(scheduler=scheduler_dec, optimizer=optimizer_dec, **kwargs)
        update_scheduler(scheduler=scheduler_enc, optimizer=optimizer_enc, **kwargs)

        log_training_epoch(
            optimizer=optimizer_dec,
            best_val_loss=best_val_loss,
            best_val_selbo=best_val_selbo,
            epoch=epoch,
            train_loss=train_loss,
            train_recon=train_recon,
            train_selbo=train_selbo,
            val_loss=val_loss,
            val_recon=val_recon,
            val_selbo=val_selbo,
            vae=vae,
            beta=1.0,
        )

        early_stopping = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            log.info("Early stopping triggered.")
            early_stopping = True
        if val_selbo < best_val_selbo:
            best_val_selbo = val_selbo

        lm_val, lm_train, lm_test = 0.0, 0.0, 0.0

        epoch_mod = epoch % 50 == 0

        if early_stopping or epoch_mod:
            kwargs = {
                "model": vae,
                "device": device,
                "input_dim": input_dim,
                "cnn": cnn,
            }

            lm_val = estimate_log_marginal(data_loader=validation_loader, **kwargs)
            lm_train = estimate_log_marginal(data_loader=train_loader, **kwargs)
            lm_test = estimate_log_marginal(data_loader=test_loader, **kwargs)
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
            beta=1.0,
        )

        if (epoch % 10 == 0) or early_stopping:
            df_stats.to_csv(pathlib.Path(model_path) / f"{file_name}.csv")
        if early_stopping:
            break


def run_experiment(iw_samples, path):
    n_layers = 4
    geometry = "flat"
    latent_dim_factor = 0.2

    train_loader, validation_loader, test_loader, dim = get_loaders()

    vae = create_vae_model(
        dim=dim,
        n_layers=n_layers,
        geometry=geometry,
        latent_dim_factor=latent_dim_factor,
    )
    log.info(f"Running iw_samples: {iw_samples}.")
    log.info(f"Save model as {path}.")

    model_name = f"sep_iwae_{iw_samples}_plog_k_500"

    train_vae(
        vae=vae,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        dim=dim,
        model_path=path,
        file_name=model_name,
        iw_samples=iw_samples,
        gamma=0.25,
        plateau_patience=7,
        patience=15,
        epochs=400,
        scheduler_type="plateau",
    )

    model_save_path = path / (model_name + ".pth")
    torch.save(vae.state_dict(), model_save_path)


def google_stuff() -> pathlib.Path:
    try:
        from google.colab import drive

        log.info("Running on Google Colab.")
        save_path = "/content/drive/My Drive/thesis/data/"
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        return pathlib.Path(save_path)

    except ImportError:
        log.info("Not running on Google Colab.")
        path = "/Users/joachim/Library/Mobile Documents/com~apple~CloudDocs/thesis/data"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        return pathlib.Path(path)


def main():
    path = google_stuff()

    for x in list(
        [
            3,
            10,
        ]
    ):
        run_experiment(iw_samples=x, path=path)


if __name__ == "__main__":
    main()
