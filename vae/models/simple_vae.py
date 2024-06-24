import time
import logging
import pathlib
import itertools
import sys
import typing

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure

import numpy as np
import pandas as pd

from vae.data.image_data import load_mnist, load_emnist, load_fashion_mnist
from vae.utils import exception_hook, model_path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class Encoder(nn.Module):
    def __init__(
        self,
        activation: nn.Module,
        dimension: pd.DataFrame,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super(Encoder, self).__init__()
        self.activation = activation
        assert n_layers >= 2
        self.dimension = dimension

        self.hidden_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for i in range(n_layers - 1):
            in_dim = dimension.loc[i, "encoder_in_dim"]
            out_dim = dimension.loc[i, "encoder_out_dim"]
            self.hidden_layers.append(nn.Linear(in_dim, out_dim))
            self.dropout_layers.append(nn.Dropout(dropout))

        last_idx = dimension.shape[0] - 1
        in_dim = dimension.loc[last_idx, "encoder_in_dim"]
        out_dim = dimension.loc[last_idx, "encoder_out_dim"]

        self.fc_mu = nn.Linear(in_dim, out_dim)
        self.fc_logvar = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        for idx, layer in enumerate(self.hidden_layers):
            x = self.activation(layer(x))
            x = self.dropout_layers[idx](x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        activation: nn.Module,
        dimension: pd.DataFrame,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super(Decoder, self).__init__()
        self.activation = activation
        assert n_layers >= 1
        self.n_hidden_layers = n_layers
        self.dimension = dimension

        self.hidden_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for i in range(n_layers):
            input_dim = dimension.loc[i, "decoder_in_dim"]
            output_dim = dimension.loc[i, "decoder_out_dim"]
            self.hidden_layers.append(nn.Linear(input_dim, output_dim))
            if i < n_layers - 1:
                self.dropout_layers.append(nn.Dropout(dropout))

    def forward(self, h):
        for i in range(self.n_hidden_layers - 1):
            layer = self.hidden_layers[i]
            h = self.activation(layer(h))
            h = self.dropout_layers[i](h)
        x_recon = torch.sigmoid(self.hidden_layers[-1](h))
        return x_recon


class VAE(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        input_dim: int,
        activation: str = "relu",
        n_layers: int = 2,
        geometry: str = "flat",
        dropout: float = 0.0,
    ):
        super(VAE, self).__init__()
        self.geometry = geometry

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2)
        elif activation == "mish":
            self.activation = nn.Mish()
        elif activation == "prelu":
            self.activation = nn.PReLU()

        self.dimension = self.generate_dimension(
            latent_dim=latent_dim,
            input_dim=input_dim,
            n_hidden_layers=n_layers,
            geometry=geometry,
            hidden_dim=hidden_dim,
        )

        self.encoder = Encoder(
            activation=self.activation,
            n_layers=n_layers,
            dimension=self.dimension,
            dropout=dropout,
        )
        self.decoder = Decoder(
            activation=self.activation,
            n_layers=n_layers,
            dimension=self.dimension,
            dropout=dropout,
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def generate_dimension(
        self,
        latent_dim: int,
        input_dim: int,
        n_hidden_layers: int,
        hidden_dim: int,
        geometry: str,
    ) -> pd.DataFrame:

        def reverse_column(df, column):
            return df[column].iloc[::-1].reset_index(drop=True)

        geo_hidden_dim = hidden_dim / n_hidden_layers

        if geometry == "geo":
            dimension = pd.DataFrame(
                index=list(range(n_hidden_layers)), data={"geo": int(geo_hidden_dim)}
            )
            dimension.index.name = "layer"
            dimension = dimension.reset_index()
            dimension.layer = dimension.layer + 1

            # ENCODER
            dimension["encoder_in_dim"] = dimension.geo * dimension.layer
            dimension["encoder_in_dim"] = reverse_column(dimension, "encoder_in_dim")
            dimension.loc[0, "encoder_in_dim"] = input_dim
            dimension["encoder_out_dim"] = dimension.encoder_in_dim.shift(-1)
            dimension.loc[dimension.shape[0] - 1, "encoder_out_dim"] = latent_dim
            dimension["encoder_out_dim"] = dimension.encoder_out_dim.astype(int)
            dimension = dimension.drop(["geo", "layer"], axis=1)

            # DECODER
            dimension["decoder_in_dim"] = reverse_column(dimension, "encoder_out_dim")
            dimension["decoder_out_dim"] = reverse_column(dimension, "encoder_in_dim")

        elif geometry == "flat":
            dimension = pd.DataFrame(
                index=list(range(n_hidden_layers)),
                data={
                    "encoder_in_dim": hidden_dim,
                    "encoder_out_dim": hidden_dim,
                    "decoder_in_dim": hidden_dim,
                    "decoder_out_dim": hidden_dim,
                },
            )
            dimension.index.name = "layer"

            dimension.loc[0, "encoder_in_dim"] = input_dim
            dimension.loc[dimension.shape[0] - 1, "encoder_out_dim"] = latent_dim
            dimension.loc[0, "decoder_in_dim"] = latent_dim
            dimension.loc[dimension.shape[0] - 1, "decoder_out_dim"] = input_dim
        return dimension


def loss_function(x_recon, x, mu, logvar) -> float:
    BCE = nn.functional.binary_cross_entropy(x_recon, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD + BCE


def mse_loss_function(
    x_recon,
    x,
) -> float:
    MSE = F.mse_loss(x_recon, x, reduction="sum")
    return MSE


def ssim_loss_function(x_recon, x) -> float:
    image_shape = (1, 28, 28)

    x_geo = x.view(-1, *image_shape)
    x_recon_geo = x_recon.view(-1, *image_shape)
    ssim_value = structural_similarity_index_measure(x_recon_geo, x_geo)
    return ssim_value


def run_model(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    activation: str = "relu",
    n_layers: int = 2,
    geometry: str = "flat",
    dropout: float = 0.0,
) -> typing.Tuple[VAE, pd.DataFrame]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = VAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        n_layers=n_layers,
        activation=activation,
        geometry=geometry,
        dropout=dropout,
    ).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    stats = {
        "train_loss": [],
        "train_mse_loss": [],
        "train_ssim": [],
        "validation_loss": [],
        "validation_mse_loss": [],
        "validation_ssim": [],
        "test_loss": [],
        "test_mse_loss": [],
        "test_ssim": [],
    }

    vae.train()
    for epoch in range(epochs):

        train_loss, train_mse_loss, train_ssim = 0, 0, 0

        for _, (data, _) in enumerate(train_loader):
            train_loss, train_mse_loss, train_ssim = training_step(
                input_dim=input_dim,
                device=device,
                vae=vae,
                optimizer=optimizer,
                train_loss=train_loss,
                train_mse_loss=train_mse_loss,
                train_ssim=train_ssim,
                data=data,
            )

        vae.eval()
        validation_loss, validation_mse_loss, validation_ssim = 0, 0, 0
        with torch.no_grad():
            for data, _ in validation_loader:
                data = data.view(-1, input_dim).to(device)
                x_recon, mu, logvar = vae(data)
                loss = loss_function(x_recon, data, mu, logvar).to(device)
                mloss = mse_loss_function(x_recon, data).to(device)
                ssim = ssim_loss_function(x_recon, data).to(device)
                validation_loss += loss.item()
                validation_mse_loss += mloss.item()
                validation_ssim += ssim.item()

        test_loss, test_mse_loss, test_ssim = 0, 0, 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.view(-1, input_dim).to(device)
                x_recon, mu, logvar = vae(data)
                loss = loss_function(x_recon, data, mu, logvar).to(device)
                mloss = mse_loss_function(x_recon, data).to(device)
                ssim = ssim_loss_function(x_recon, data).to(device)
                test_loss += loss.item()
                test_mse_loss += mloss.item()
                test_ssim += ssim.item()

        add_epoch_stats(
            stats=stats,
            train_loader=train_loader,
            validation_loader=validation_loader,
            test_loader=test_loader,
            train_loss=train_loss,
            train_mse_loss=train_mse_loss,
            train_ssim=train_ssim,
            validation_loss=validation_loss,
            validation_mse_loss=validation_mse_loss,
            validation_ssim=validation_ssim,
            test_loss=test_loss,
            test_mse_loss=test_mse_loss,
            test_ssim=test_ssim,
        )

        if epoch > 1:
            pct_val = (
                stats["validation_loss"][-1] - stats["validation_loss"][-2]
            ) / stats["validation_loss"][-2]
            pct_val = pct_val * 100

            log.info(
                f"Epoch {epoch} |"
                f"Train Loss {stats['train_loss'][-1]:.2f} | "
                f"Val Loss {stats['validation_loss'][-1]:.2f} | "
                f"Train MSE Loss {stats['train_mse_loss'][-1]:.2f} | "
                f"Val MSE Loss {stats['validation_mse_loss'][-1]:.2f} | "
                f"Train SSIM {stats['train_ssim'][-1]:.2f} | "
                f"Val SSIM {stats['validation_ssim'][-1]:.2f} |"
                f"Val Loss Chg {pct_val:.2f} %"
            )

    stats = pd.DataFrame(stats)

    return vae, stats


def training_step(
    input_dim: int,
    device: str,
    vae: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loss: float,
    train_mse_loss: float,
    train_ssim: float,
    data: typing.Any,
):
    data = data.view(-1, input_dim).to(device)
    optimizer.zero_grad()
    x_recon, mu, logvar = vae(data)
    loss = loss_function(x_recon, data, mu, logvar).to(device)
    mloss = mse_loss_function(x_recon, data).to(device)
    ssim = ssim_loss_function(x_recon, data).to(device)
    loss.backward()
    train_loss += loss.item()
    train_mse_loss += mloss.item()
    train_ssim += ssim.item()
    optimizer.step()

    return train_loss, train_mse_loss, train_ssim


def add_epoch_stats(
    stats: dict,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    train_loss: float,
    train_mse_loss: float,
    train_ssim: float,
    validation_loss: float,
    validation_mse_loss: float,
    validation_ssim: float,
    test_loss: float,
    test_mse_loss: float,
    test_ssim: float,
):
    stats["train_loss"].append(train_loss / len(train_loader.dataset))
    stats["train_mse_loss"].append(train_mse_loss / len(train_loader.dataset))
    stats["train_ssim"].append(train_ssim / len(train_loader.dataset))

    stats["validation_loss"].append(validation_loss / len(validation_loader.dataset))
    stats["validation_mse_loss"].append(
        validation_mse_loss / len(validation_loader.dataset)
    )
    stats["validation_ssim"].append(validation_ssim / len(validation_loader.dataset))

    stats["test_loss"].append(test_loss / len(test_loader.dataset))
    stats["test_mse_loss"].append(test_mse_loss / len(test_loader.dataset))
    stats["test_ssim"].append(test_ssim / len(test_loader.dataset))


sys.excepthook = exception_hook


def run(
    data_set="mnist",
    activation="relu",
    n_layers=2,
    geometry="flat",
    latent_dim_factor=0.1,
    batch_size=128,
    dropout: float = 0.0,
):
    learning_rate = 1e-3
    epochs = 100

    if data_set == "mnist":
        load_fn = load_mnist
    elif data_set == "emnist":
        load_fn = load_emnist
    elif data_set == "fashion_mnist":
        load_fn = load_fashion_mnist
    else:
        raise ValueError("Invalid dataset")

    train_loader, validation_loader, test_loader, dim = load_fn(
        batch_size=batch_size,
    )

    hidden_dim = int(np.prod(dim))
    latent_dim = int(np.prod(dim) * latent_dim_factor)

    dim = np.prod(dim)

    _, stats = run_model(
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        input_dim=dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        learning_rate=learning_rate,
        epochs=epochs,
        geometry=geometry,
        n_layers=n_layers,
        activation=activation,
        dropout=dropout,
    )
    return stats


def parameter_to_string(
    data_set="mnist",
    activation="relu",
    n_layers=2,
    geometry="flat",
    latent_dim_factor=0.1,
    batch_size=128,
    dropout: float = 0.0,
) -> str:
    return (
        f"data_set-{data_set}-"
        f"activation-{activation}-"
        f"n_layers-{n_layers}-"
        f"geometry-{geometry}-"
        f"latent_dim_factor-{latent_dim_factor}-"
        f"batch_size-{batch_size}-"
        f"dropout-{dropout}"
    )


def print_parameter(
    data_set="mnist",
    activation="relu",
    n_layers=2,
    geometry="flat",
    latent_dim_factor=0.1,
    batch_size=128,
    dropout: float = 0.0,
):
    ret = {
        "data_set": data_set,
        "activation": activation,
        "n_layers": n_layers,
        "geometry": geometry,
        "latent_dim_factor": latent_dim_factor,
        "batch_size": batch_size,
        "dropout": dropout,
    }
    ret_str = ""
    for key, value in ret.items():
        ret_str += f"{key}: {value}\n"
    print(ret_str)


def string_to_parameter(s: str) -> typing.Tuple[str, str, int, str, float, int]:
    parts = s.split("-")
    data_set = parts[1]
    dropout = float(parts[3])
    activation = parts[5]
    n_layers = int(parts[7])
    geometry = parts[9]
    latent_dim_factor = float(parts[11])
    batch_size = int(parts[13])
    return (
        data_set,
        activation,
        n_layers,
        geometry,
        latent_dim_factor,
        batch_size,
        dropout,
    )


def main():
    data_sets = ["mnist"]

    dropouts = [0.0, 0.3]
    activations = [
        "prelu",
        "relu",
        "mish",
    ]
    n_layerss = [2, 3, 4]
    geometrys = ["geo", "flat"]
    latent_dim_factors = [0.1, 0.025]
    batch_sizes = [64, 128]

    parameters = list(
        itertools.product(
            data_sets,
            dropouts,
            activations,
            n_layerss,
            geometrys,
            latent_dim_factors,
            batch_sizes,
        )
    )
    log.info(f"Running {len(parameters)} experiments")

    for parameter in parameters:
        (
            data_set,
            dropout,
            activation,
            n_layers,
            geometry,
            latent_dim_factor,
            batch_size,
        ) = parameter
        print_parameter(
            data_set=data_set,
            dropout=dropout,
            activation=activation,
            n_layers=n_layers,
            geometry=geometry,
            latent_dim_factor=latent_dim_factor,
            batch_size=batch_size,
        )

        start_time = time.time()

        stats = run(
            data_set=data_set,
            activation=activation,
            n_layers=n_layers,
            geometry=geometry,
            latent_dim_factor=latent_dim_factor,
            batch_size=batch_size,
            dropout=dropout,
        )
        file_name = parameter_to_string(
            data_set=data_set,
            activation=activation,
            n_layers=n_layers,
            geometry=geometry,
            latent_dim_factor=latent_dim_factor,
            batch_size=batch_size,
            dropout=dropout,
        )

        path = model_path() / "experiment" / pathlib.Path(str(file_name) + ".csv")
        stats.to_csv(path)

        end_time = time.time()
        log.info(f"Experiment took {end_time - start_time} seconds.")


if __name__ == "__main__":
    main()
