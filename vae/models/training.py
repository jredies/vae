import sys
import functools
import pathlib
import typing
import logging

import numpy as np
import pandas as pd

import torch
from torch.nn import functional as F
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vae.models.simple_vae import create_vae_model
from vae.data.image_data import load_mnist, load_emnist, load_fashion_mnist
from vae.utils import exception_hook, model_path
from vae.models.loss import standard_loss, reconstruction_loss
from vae.models.loss import iwae_loss_fast_cnn, iwae_loss_fast
from vae.models.noise import add_gaussian_noise, add_salt_and_pepper_noise
from vae.models.utils import select_device


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def get_loaders(
    rotation: int = 0,
    translate: float = 0.0,
    batch_size: int = 128,
) -> typing.Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    train_loader, validation_loader, test_loader, dimensionality = load_mnist(
        rotation=rotation,
        translate=translate,
        batch_size=batch_size,
    )
    return train_loader, validation_loader, test_loader, dimensionality


def estimate_log_marginal(
    model,
    data_loader,
    device,
    input_dim: int,
    num_samples=10,
    cnn=False,
) -> float:
    log_weights = []
    with torch.no_grad():
        for x, _ in data_loader:
            x = get_view(device=device, input_dim=input_dim, x=x, cnn=cnn)
            if cnn:
                log_weights.append(
                    iwae_loss_fast_cnn(model=model, x=x, num_samples=num_samples)
                )
            else:
                log_weights.append(
                    iwae_loss_fast(model=model, x=x, num_samples=num_samples)
                )

    return np.array([x.item() for x in log_weights]).mean()


def get_view(device, input_dim, x, cnn=False):
    if cnn:
        x = x.view(-1, 1, 28, 28).to(device)
    else:
        x = x.view(-1, input_dim).to(device)
    return x


def training_step(
    input_dim: int,
    device: str,
    vae: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loss: float,
    train_recon: float,
    data: typing.Any,
    gaussian_noise: float = 0.0,
    salt_and_pepper_noise: float = 0.0,
    clip_gradient: bool = False,
    norm_gradient: bool = False,
    beta: float = 1.0,
    loss_fn: typing.Callable = standard_loss,
    cnn=False,
) -> typing.Tuple[float, float]:

    data = get_view(device=device, input_dim=input_dim, x=data, cnn=cnn)

    if gaussian_noise > 0.0:
        data = add_gaussian_noise(data, noise_factor=gaussian_noise)
    if salt_and_pepper_noise > 0.0:
        data = add_salt_and_pepper_noise(data, noise_factor=salt_and_pepper_noise)

    optimizer.zero_grad()
    x_recon, mu, logvar = vae(data)
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
    loss.backward()

    train_loss += loss.item()
    train_recon += recon.item()

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

    return train_loss, train_recon


def calculate_stats(
    vae: nn.Module,
    loader: DataLoader,
    device: str,
    input_dim: int,
    loss_fn: typing.Callable = standard_loss,
    cnn=False,
) -> typing.Tuple[float, float]:
    ret_loss, ret_recon = 0.0, 0.0

    mini_batches = 0

    with torch.no_grad():
        for data, _ in loader:
            data = get_view(device=device, input_dim=input_dim, x=data, cnn=cnn)
            x_recon, mu, logvar = vae(data)
            loss = loss_fn(
                x_recon=x_recon,
                x=data,
                mu=mu,
                logvar=logvar,
                model=vae,
                cnn=cnn,
            ).to(device)
            recon = reconstruction_loss(x_recon=x_recon, x=data).to(device)
            ret_loss += loss.item()
            ret_recon += recon.item()
            mini_batches += 1

    ret_loss = ret_loss / mini_batches
    ret_recon = ret_recon / mini_batches

    return ret_loss, ret_recon


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
    loss_type: str = "standard",
    iw_samples: int = 5,
    cnn: bool = False,
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

    loss_fn = standard_loss
    if loss_type == "standard":
        loss_fn = standard_loss
    elif loss_type == "iwae":
        if cnn:
            loss_fn = functools.partial(iwae_loss_fast, num_samples=iw_samples)
        else:
            loss_fn = functools.partial(iwae_loss_fast_cnn, num_samples=iw_samples)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        vae.train()
        train_loss, train_recon, mini_batches = 0.0, 0.0, 0

        for _, (data, _) in enumerate(train_loader):
            train_loss, train_recon = training_step(
                input_dim=input_dim,
                device=device,
                vae=vae,
                optimizer=optimizer,
                train_loss=train_loss,
                train_recon=train_recon,
                data=data,
                gaussian_noise=gaussian_noise,
                salt_and_pepper_noise=salt_and_pepper_noise,
                norm_gradient=norm_gradient,
                clip_gradient=clip_gradient,
                loss_fn=loss_fn,
                cnn=cnn,
            )
            mini_batches += 1

        train_loss /= mini_batches
        train_recon /= mini_batches

        vae.eval()
        val_loss, val_recon = calculate_stats(
            vae=vae,
            loader=validation_loader,
            device=device,
            input_dim=input_dim,
            loss_fn=loss_fn,
            cnn=cnn,
        )
        test_loss, test_recon = calculate_stats(
            vae=vae,
            loader=test_loader,
            device=device,
            input_dim=input_dim,
            loss_fn=loss_fn,
            cnn=cnn,
        )

        update_scheduler(
            scheduler_type=scheduler_type,
            gamma=gamma,
            scheduler=scheduler,
            val_loss=val_loss,
        )

        log_training_epoch(
            optimizer=optimizer,
            best_val_loss=best_val_loss,
            epoch=epoch,
            train_loss=train_loss,
            train_recon=train_recon,
            val_loss=val_loss,
            val_recon=val_recon,
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

        lm_val, lm_train, lm_test = 0.0, 0.0, 0.0

        # if (epoch % 100 == 0) or early_stopping:
        #     lm_val = estimate_log_marginal(
        #         model=vae,
        #         data_loader=validation_loader,
        #         device=device,
        #         input_dim=input_dim,
        #         cnn=cnn,
        #     )
        #     lm_train = estimate_log_marginal(
        #         model=vae,
        #         data_loader=train_loader,
        #         device=device,
        #         input_dim=input_dim,
        #         cnn=cnn,
        #     )
        #     lm_test = estimate_log_marginal(
        #         model=vae,
        #         data_loader=test_loader,
        #         device=device,
        #         input_dim=input_dim,
        #         cnn=cnn,
        #     )

        write_all_stats(
            writer=writer,
            df_stats=df_stats,
            epoch=epoch,
            train_loss=train_loss,
            train_lm=lm_train,
            train_recon=train_recon,
            val_loss=val_loss,
            val_lm=lm_val,
            val_recon=val_recon,
            test_loss=test_loss,
            test_lm=lm_test,
            test_recon=test_recon,
            best_val_loss=best_val_loss,
        )

        if early_stopping:
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
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1_000_0000,
            gamma=0.99,
        )

    return scheduler


def write_all_stats(
    writer: SummaryWriter,
    df_stats: pd.DataFrame,
    epoch: int,
    train_loss: float,
    train_lm: float,
    train_recon: float,
    val_loss: float,
    val_lm: float,
    val_recon: float,
    test_loss: float,
    test_lm: float,
    test_recon: float,
    best_val_loss: float,
):
    write_stats("train_loss", train_loss, epoch, writer, df_stats)
    write_stats("val_loss", val_loss, epoch, writer, df_stats)
    write_stats("test_loss", test_loss, epoch, writer, df_stats)

    # write_stats("train_lm", train_lm, epoch, writer, df_stats)
    # write_stats("val_lm", val_lm, epoch, writer, df_stats)
    # write_stats("test_lm", test_lm, epoch, writer, df_stats)

    write_stats("train_recon", train_recon, epoch, writer, df_stats)
    write_stats("val_recon", val_recon, epoch, writer, df_stats)
    write_stats("test_recon", test_recon, epoch, writer, df_stats)

    write_stats(
        "best_val_loss-val_loss", best_val_loss - val_loss, epoch, writer, df_stats
    )


def log_training_epoch(
    optimizer,
    best_val_loss,
    epoch,
    train_loss,
    val_loss,
    train_recon,
    val_recon,
):
    formatted_epoch = str(epoch).zfill(3)

    output_string = (
        f" Epoch {formatted_epoch} |"
        f" LR {optimizer.param_groups[0]['lr']:.7f} |"
        f" Tr Loss {train_loss:.4f} |"
        # f" Tr LM {train_lm:.4f} |"
        f" Tr Recon {train_recon:.6f} |"
        f" Val Loss {val_loss:.4f} |"
        # f" Val LM {val_lm:.4f} |"
        f" Val Recon {val_recon:.6f} |"
    )
    if epoch >= 1:
        diff_val = val_loss - best_val_loss
        output_string += f" Val now - best {diff_val:.6f}"

    log.info(output_string)


sys.excepthook = exception_hook


def main():
    path = "logs/aug"

    train_loader, validation_loader, test_loader, dim = get_loaders()

    latent_factor = 0.05
    layer = 3


if __name__ == "__main__":
    main()