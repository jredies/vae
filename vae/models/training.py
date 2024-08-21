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


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


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
    num_samples=100,
    cnn=False,
) -> float:
    log_weights = []
    with torch.no_grad():
        for x, _ in data_loader:
            x = get_view(device=device, input_dim=input_dim, x=x, cnn=cnn)
            if cnn:
                log_weights.append(
                    iwae_loss_fast_cnn(
                        model=model,
                        x=x,
                        num_samples=num_samples,
                    )
                )
            else:
                log_weights.append(
                    iwae_loss_fast(
                        model=model,
                        x=x,
                        num_samples=num_samples,
                    )
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
    train_selbo: float,
    data: typing.Any,
    gaussian_noise: float = 0.0,
    salt_and_pepper_noise: float = 0.0,
    latent_noise: float = 0.0,
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


def calculate_stats(
    vae: nn.Module,
    loader: DataLoader,
    device: str,
    input_dim: int,
    loss_fn: typing.Callable = standard_loss,
    cnn=False,
) -> typing.Tuple[float, float]:
    ret_loss, ret_recon, ret_selbo = 0.0, 0.0, 0.0

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
            selbo = standard_loss(
                x_recon=x_recon,
                x=data,
                mu=mu,
                logvar=logvar,
                cnn=cnn,
            ).to(device)

            ret_loss += loss.item()
            ret_recon += recon.item()
            ret_selbo += selbo.item()
            mini_batches += 1

    ret_loss = ret_loss / mini_batches
    ret_recon = ret_recon / mini_batches
    ret_selbo = ret_selbo / mini_batches

    return ret_loss, ret_recon, ret_selbo


def write_stats(
    label: str,
    value: float,
    epoch: int,
    writer: SummaryWriter,
    df_stats: pd.DataFrame,
):
    writer.add_scalar(label, value, epoch)
    df_stats.loc[epoch, label] = value


def get_learning_rate(epoch):
    initial_lr = 0.001
    decay_factor = 10 ** (-1 / 7)

    # Define the epoch boundaries for each stage
    stage_epochs = [3**i for i in range(8)]
    epoch_boundaries = [sum(stage_epochs[: i + 1]) for i in range(len(stage_epochs))]

    # Find the current stage based on the epoch
    stage = next(i for i, boundary in enumerate(epoch_boundaries) if epoch < boundary)

    # Calculate the learning rate for the current stage
    learning_rate = initial_lr * (decay_factor**stage)

    return learning_rate


def update_learning_rate(optimizer, epoch, annealing_start, annealing_end):
    # Get the new learning rate based on the epoch
    if epoch < annealing_start:
        new_lr = get_learning_rate(epoch)
    elif epoch <= annealing_end:
        new_lr = get_learning_rate(epoch - annealing_start)
    elif epoch >= annealing_end:
        new_lr = get_learning_rate(epoch - annealing_end)
    else:
        raise ValueError("Epoch out of bounds")

    # Update the learning rate in the optimizer
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


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

    optimizer = optim.Adam(
        vae.parameters(),
        lr=base_learning_rate,
        eps=1e-4,
    )

    scheduler = initialize_scheduler(
        scheduler_type=scheduler_type,
        plateau_patience=plateau_patience,
        step_size=step_size,
        gamma=gamma,
        optimizer=optimizer,
    )

    loss_fn = set_loss(loss_type, iw_samples=iw_samples, cnn=cnn)

    best_val_loss = float("inf")
    best_val_selbo = float("inf")
    patience_counter = 0

    init_weights(vae)

    for epoch in range(epochs):
        vae.train()
        train_loss, train_recon, train_selbo, mini_batches = 0.0, 0.0, 0.0, 0
        beta = calculate_beta(
            annealing_start=annealing_start,
            annealing_end=annealing_end,
            annealing_type=annealing_type,
            epoch=epoch,
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
                latent_noise=latent_noise,
                norm_gradient=norm_gradient,
                clip_gradient=clip_gradient,
                loss_fn=loss_fn,
                cnn=cnn,
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
            loss_fn=loss_fn,
            cnn=cnn,
        )
        test_loss, test_recon, test_selbo = calculate_stats(
            vae=vae,
            loader=test_loader,
            device=device,
            input_dim=input_dim,
            loss_fn=loss_fn,
            cnn=cnn,
        )

        if epoch > annealing_end:
            update_scheduler(
                scheduler_type=scheduler_type,
                gamma=gamma,
                scheduler=scheduler,
                val_loss=val_loss,
                epoch=epoch,
                optimizer=optimizer,
                annealing_start=annealing_start,
                annealing_end=annealing_end,
            )

        log_training_epoch(
            optimizer=optimizer,
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
            beta=beta,
        )

        early_stopping = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        elif epoch < annealing_end:
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
            lm_val = estimate_log_marginal(
                model=vae,
                data_loader=validation_loader,
                device=device,
                input_dim=input_dim,
                cnn=cnn,
            )
            lm_train = estimate_log_marginal(
                model=vae,
                data_loader=train_loader,
                device=device,
                input_dim=input_dim,
                cnn=cnn,
            )
            lm_test = estimate_log_marginal(
                model=vae,
                data_loader=test_loader,
                device=device,
                input_dim=input_dim,
                cnn=cnn,
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


def set_loss(loss_type, iw_samples, cnn):
    if loss_type == "standard":
        loss_fn = standard_loss
    elif loss_type == "reconstruction":
        loss_fn = reconstruction_loss
    elif loss_type == "iwae":
        if cnn:
            loss_fn = functools.partial(iwae_loss_fast_cnn, num_samples=iw_samples)
        else:
            loss_fn = functools.partial(iwae_loss_fast, num_samples=iw_samples)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    return loss_fn


def calculate_beta(
    epoch: int,
    annealing_start: int,
    annealing_end: int,
    annealing_type: str,
) -> float:
    if annealing_start == 0 and annealing_end == 0:
        return 1.0
    elif (epoch >= annealing_start) and (epoch <= annealing_end):
        if annealing_type == "linear":
            beta = (epoch - annealing_start) / (annealing_end - annealing_start)
        elif annealing_type == "step":
            beta = 0.0
    elif epoch < annealing_start:
        beta = 0.0
    else:
        beta = 1.0

    # beta = max(beta, 0.05)

    return beta


def update_scheduler(
    scheduler_type: str,
    gamma: float,
    scheduler,
    val_loss: float,
    epoch: int,
    optimizer,
    annealing_start: int,
    annealing_end: int,
):
    if scheduler_type == "paper":
        update_learning_rate(optimizer, epoch, annealing_start, annealing_end)
    if gamma < 1.0:
        if scheduler_type == "plateau":
            scheduler.step(val_loss)
        elif (scheduler_type == "step") or (scheduler_type == "exponential"):
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
    train_selbo: float,
    val_loss: float,
    val_lm: float,
    val_recon: float,
    val_selbo: float,
    test_loss: float,
    test_lm: float,
    test_recon: float,
    test_selbo: float,
    best_val_loss: float,
    beta: float,
):
    kwargs = {"epoch": epoch, "writer": writer, "df_stats": df_stats}

    write_stats("train_loss", train_loss, **kwargs)
    write_stats("val_loss", val_loss, **kwargs)
    write_stats("test_loss", test_loss, **kwargs)

    write_stats("train_lm", train_lm, **kwargs)
    write_stats("val_lm", val_lm, **kwargs)
    write_stats("test_lm", test_lm, **kwargs)

    write_stats("train_recon", train_recon, **kwargs)
    write_stats("val_recon", val_recon, **kwargs)
    write_stats("test_recon", test_recon, **kwargs)

    write_stats("train_selbo", train_selbo, **kwargs)
    write_stats("val_selbo", val_selbo, **kwargs)
    write_stats("test_selbo", test_selbo, **kwargs)

    write_stats("best_val_loss-val_loss", best_val_loss - val_loss, **kwargs)
    write_stats("beta", beta, **kwargs)


def log_training_epoch(
    optimizer,
    best_val_loss,
    best_val_selbo,
    epoch,
    train_loss,
    val_loss,
    train_recon,
    train_selbo,
    val_recon,
    val_selbo,
    vae,
    beta,
):
    formatted_epoch = str(epoch).zfill(3)

    output_string = (
        # f" SM {vae.spectral_norm} |"
        f" Epoch {formatted_epoch} |"
        f" Beta {beta:.2f} |"
        f" LR {optimizer.param_groups[0]['lr']:.7f} |"
        f" Tr Loss {train_loss:.2f} |"
        # f" Tr LM {train_lm:.4f} |"
        f" Tr Recon {train_recon:.2f} |"
        f" Tr SELBO {train_selbo:.2f} |"
        f" Val Loss {val_loss:.2f} |"
        # f" Val LM {val_lm:.4f} |"
        f" Val Recon {val_recon:.2f} |"
        f" Val SELBO {val_selbo:.2f} |"
    )
    if epoch >= 1:
        diff_val = val_loss - best_val_loss
        output_string += f" Val now - best {diff_val:.6f} |"

        diff_val_selbo = val_selbo - best_val_selbo
        # output_string += f" Val SELBO now - best {diff_val_selbo:.6f}"

    log.info(output_string)


sys.excepthook = exception_hook


def main():
    path = "logs/aug"

    train_loader, validation_loader, test_loader, dim = get_loaders()

    latent_factor = 0.05
    layer = 3


if __name__ == "__main__":
    main()
