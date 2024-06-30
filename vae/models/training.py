import pathlib
import typing
import logging

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vae.models.simple_vae import create_vae_model
from vae.data.image_data import load_mnist, load_emnist, load_fashion_mnist
from vae.utils import exception_hook, model_path
from vae.models.loss import loss_function, mse_loss_function
from vae.models.noise import add_gaussian_noise, add_salt_and_pepper_noise


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info(f"Using device {device}.")
    return device


def get_loaders(
    rotation: int = 0,
    translate: float = 0.0,
) -> typing.Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    train_loader, validation_loader, test_loader, dimensionality = load_mnist(
        rotation=rotation,
        translate=translate,
    )
    return train_loader, validation_loader, test_loader, dimensionality


def training_step(
    input_dim: int,
    device: str,
    vae: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loss: float,
    train_mse_loss: float,
    data: typing.Any,
    gaussian_noise: float = 0.0,
    salt_and_pepper_noise: float = 0.0,
    clip_gradient: bool = False,
    norm_gradient: bool = False,
    beta: float = 1.0,
) -> typing.Tuple[float, float]:
    data = data.view(-1, input_dim).to(device)

    if gaussian_noise > 0.0:
        data = add_gaussian_noise(data, noise_factor=gaussian_noise)
    if salt_and_pepper_noise > 0.0:
        data = add_salt_and_pepper_noise(data, noise_factor=salt_and_pepper_noise)

    optimizer.zero_grad()
    x_recon, mu, logvar = vae(data)
    loss = loss_function(
        x_recon,
        data,
        mu=mu,
        logvar=logvar,
        beta=beta,
    ).to(device)
    mloss = mse_loss_function(x_recon, data).to(device)
    loss.backward()
    train_loss += loss.item()
    train_mse_loss += mloss.item()

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

    return train_loss, train_mse_loss


def calculate_val_stats(
    vae: nn.Module,
    validation_loader: DataLoader,
    device: str,
    input_dim: int,
) -> typing.Tuple[float, float]:
    val_loss, val_mse = 0.0, 0.0
    with torch.no_grad():
        for data, _ in validation_loader:
            data = data.view(-1, input_dim).to(device)
            x_recon, mu, logvar = vae(data)
            loss = loss_function(x_recon, data, mu, logvar).to(device)
            mloss = mse_loss_function(x_recon, data).to(device)
            val_loss += loss.item()
            val_mse += mloss.item()

    n_val = len(validation_loader.dataset)
    val_loss = val_loss / n_val
    val_mse = val_mse / n_val

    return val_loss, val_mse


def calculate_test_loss(
    vae: nn.Module,
    test_loader: DataLoader,
    device: str,
    input_dim: int,
) -> typing.Tuple[float, float]:
    test_loss, test_mse = 0.0, 0.0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(-1, input_dim).to(device)
            x_recon, mu, logvar = vae(data)
            loss = loss_function(x_recon, data, mu, logvar).to(device)
            mloss = mse_loss_function(x_recon, data).to(device)
            test_loss += loss.item()
            test_mse += mloss.item()

    n_test = len(test_loader.dataset)
    test_loss = test_loss / n_test
    test_mse = test_mse / n_test

    return test_loss, test_mse


def write_stats(
    label: str,
    value: float,
    epoch: int,
    writer: SummaryWriter,
    df_stats: pd.DataFrame,
):
    writer.add_scalar(label, value, epoch)
    df_stats.loc[epoch, label] = value


def train_vae(
    model_path: str,
    vae: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    dim: np.ndarray,
    base_learning_rate: float = 1e-3,
    patience: int = 10,
    epochs: int = 200,
    scheduler_type: str = "plateau",
    plateau_patience: int = 5,
    step_size: int = 10,
    gamma: float = 1.0,
    gaussian_noise: float = 0.0,
    salt_and_pepper_noise: float = 0.0005,
    clip_gradient: bool = False,
    norm_gradient: bool = False,
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

        for _, (data, _) in enumerate(train_loader):
            train_loss, train_mse = training_step(
                input_dim=input_dim,
                device=device,
                vae=vae,
                optimizer=optimizer,
                train_loss=train_loss,
                train_mse_loss=train_mse,
                data=data,
                gaussian_noise=gaussian_noise,
                salt_and_pepper_noise=salt_and_pepper_noise,
                norm_gradient=norm_gradient,
                clip_gradient=clip_gradient,
            )

        n_train = len(train_loader.dataset)
        train_loss /= n_train
        train_mse /= n_train

        vae.eval()
        val_loss, val_mse = calculate_val_stats(
            vae=vae,
            validation_loader=validation_loader,
            device=device,
            input_dim=input_dim,
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
            train_mse=train_mse,
            val_loss=val_loss,
            val_mse=val_mse,
            test_loss=test_loss,
            test_mse=test_mse,
        )
        log_training_epoch(
            optimizer=optimizer,
            best_val_loss=best_val_loss,
            epoch=epoch,
            train_loss=train_loss,
            train_mse=train_mse,
            val_loss=val_loss,
            val_mse=val_mse,
        )

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


def update_scheduler(scheduler_type: str, gamma: float, scheduler, val_loss: float):
    if gamma < 1.0:
        if scheduler_type == "plateau":
            scheduler.step(val_loss)
        if (scheduler_type == "step") or (scheduler_type == "exponential"):
            scheduler.step()


def initialize_scheduler(
    scheduler_type: str,
    plateau_patience: int,
    step_size: int,
    gamma: float,
    optimizer,
):
    if gamma < 1.0:
        if scheduler_type == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=gamma,
                patience=plateau_patience,
            )
        if scheduler_type == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=gamma,
            )
        if scheduler_type == "exponential":
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=gamma,
            )

    return scheduler


def write_all_stats(
    writer: SummaryWriter,
    df_stats: pd.DataFrame,
    epoch: int,
    train_loss: float,
    train_mse: float,
    val_loss: float,
    val_mse: float,
    test_loss: float,
    test_mse: float,
):
    write_stats("train_loss", train_loss, epoch, writer, df_stats)
    write_stats("train_mse", train_mse, epoch, writer, df_stats)
    write_stats("val_loss", val_loss, epoch, writer, df_stats)
    write_stats("val_mse", val_mse, epoch, writer, df_stats)
    write_stats("test_loss", test_loss, epoch, writer, df_stats)
    write_stats("test_mse", test_mse, epoch, writer, df_stats)


def log_training_epoch(
    optimizer, best_val_loss, epoch, train_loss, train_mse, val_loss, val_mse
):
    formatted_epoch = str(epoch).zfill(3)

    output_string = (
        f" Epoch {formatted_epoch} |  "
        f"LR {optimizer.param_groups[0]['lr']:.7f} |"
        f" Train Loss {train_loss:.2f} |"
        f" Train MSE {train_mse:.2f} |"
        f" Validation Loss {val_loss:.2f} |"
        f" Validation MSE {val_mse:.2f} |"
    )
    if epoch >= 1:
        pct_val = (val_loss - best_val_loss) / best_val_loss
        pct_val = pct_val * 100
        output_string += f" Val improvement from best {pct_val:.2f} % "

    log.info(output_string)


def main():
    path = "logs/aug"

    train_loader, validation_loader, test_loader, dim = get_loaders()

    latent_factor = 0.05
    layer = 3

    vae = create_vae_model(
        dim=dim,
        n_layers=layer,
        geometry="flat",
        latent_dim_factor=latent_factor,
    )
    df_stats = train_vae(
        vae=vae,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        dim=dim,
        model_path=path,
        gamma=0.25,
        epochs=300,
        salt_and_pepper_noise=0.0005,
    )
    df_stats.to_csv(f"stats.csv")


if __name__ == "__main__":
    main()
