import torch
import torch.nn as nn

import pandas as pd
import numpy as np


def reverse_column(df, column):
    return df[column].iloc[::-1].reset_index(drop=True)


def geometric_dimensions(
    latent_dim: int,
    input_dim: int,
    n_hidden_layers: int,
    geo_hidden_dim: int,
) -> pd.DataFrame:
    ret = pd.DataFrame(
        index=list(range(n_hidden_layers)), data={"geo": int(geo_hidden_dim)}
    )
    ret.index.name = "layer"
    ret = ret.reset_index()
    ret.layer = ret.layer + 1

    ret["encoder_in_dim"] = ret.geo * ret.layer
    ret["encoder_in_dim"] = reverse_column(ret, "encoder_in_dim")
    ret.loc[0, "encoder_in_dim"] = input_dim
    ret["encoder_out_dim"] = ret.encoder_in_dim.shift(-1)
    ret.loc[ret.shape[0] - 1, "encoder_out_dim"] = latent_dim
    ret["encoder_out_dim"] = ret.encoder_out_dim.astype(int)
    ret = ret.drop(["geo", "layer"], axis=1)

    ret["decoder_in_dim"] = reverse_column(ret, "encoder_out_dim")
    ret["decoder_out_dim"] = reverse_column(ret, "encoder_in_dim")
    return ret


def flat_dimensions(
    latent_dim: int,
    input_dim: int,
    n_hidden_layers: int,
    hidden_dim: int,
) -> pd.DataFrame:
    ret = pd.DataFrame(
        index=list(range(n_hidden_layers)),
        data={
            "encoder_in_dim": hidden_dim,
            "encoder_out_dim": hidden_dim,
            "decoder_in_dim": hidden_dim,
            "decoder_out_dim": hidden_dim,
        },
    )
    ret.index.name = "layer"

    ret.loc[0, "encoder_in_dim"] = input_dim
    ret.loc[ret.shape[0] - 1, "encoder_out_dim"] = latent_dim
    ret.loc[0, "decoder_in_dim"] = latent_dim
    ret.loc[ret.shape[0] - 1, "decoder_out_dim"] = input_dim
    return ret


def generate_dimension(
    latent_dim: int,
    input_dim: int,
    n_hidden_layers: int,
    hidden_dim: int,
    geometry: str,
) -> pd.DataFrame:

    geo_hidden_dim = hidden_dim / n_hidden_layers

    if geometry == "geo":
        return geometric_dimensions(
            latent_dim,
            input_dim,
            n_hidden_layers,
            geo_hidden_dim,
        )

    elif geometry == "flat":
        return flat_dimensions(
            latent_dim,
            input_dim,
            n_hidden_layers,
            hidden_dim,
        )


class VariationalDropout(nn.Module):
    def __init__(self, input_dim, initial_dropout_rate=0.5):
        super(VariationalDropout, self).__init__()
        self.input_dim = input_dim
        self.dropout_rate = nn.Parameter(torch.Tensor(input_dim))
        nn.init.constant_(self.dropout_rate, initial_dropout_rate)

    def forward(self, x):
        if self.training:
            # Sample from a Bernoulli distribution with the learned dropout rates
            epsilon = torch.rand_like(x)
            mask = (epsilon > self.dropout_rate).float()
            return x * mask / (1 - self.dropout_rate)
        else:
            # During inference, use the expected value
            return x * (1 - self.dropout_rate)


class Encoder(nn.Module):
    def __init__(
        self,
        activation: nn.Module,
        dimension: pd.DataFrame,
        n_layers: int = 2,
        dropout: float = 0.3,
        batch_norm: bool = True,
        drop_type: str = "standard",
    ):
        super(Encoder, self).__init__()
        self.activation = activation
        assert n_layers >= 1
        self.dimension = dimension
        self.batch_norm = batch_norm

        self.layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()

        for i in range(n_layers - 1):
            in_dim = dimension.loc[i, "encoder_in_dim"]
            out_dim = dimension.loc[i, "encoder_out_dim"]
            self.layers.append(nn.Linear(in_dim, out_dim))
            if drop_type == "standard":
                self.dropout_layers.append(nn.Dropout(dropout))
            elif drop_type == "variational":
                self.dropout_layers.append(
                    VariationalDropout(out_dim, initial_dropout_rate=dropout)
                )
            else:
                raise ValueError(f"Unknown dropout type: {drop_type}")
            if self.batch_norm:
                self.batch_norm_layers.append(nn.BatchNorm1d(out_dim))

        assert len(self.layers) == n_layers - 1

        last_idx = dimension.shape[0] - 1
        in_dim = dimension.loc[last_idx, "encoder_in_dim"]
        out_dim = dimension.loc[last_idx, "encoder_out_dim"]

        self.fc_mu = nn.Linear(in_dim, out_dim)
        self.fc_logvar = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if self.batch_norm:
                x = self.batch_norm_layers[idx](x)
            x = self.activation(x)
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
        dropout: float = 0.3,
        batch_norm: bool = True,
        drop_type: str = "standard",
    ):
        super(Decoder, self).__init__()
        self.activation = activation
        assert n_layers >= 1
        self.n_layers = n_layers
        self.dimension = dimension
        self.batch_norm = batch_norm

        self.layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()

        for i in range(n_layers):
            input_dim = dimension.loc[i, "decoder_in_dim"]
            output_dim = dimension.loc[i, "decoder_out_dim"]
            self.layers.append(nn.Linear(input_dim, output_dim))
            if i < n_layers - 1:
                if self.batch_norm:
                    self.batch_norm_layers.append(nn.BatchNorm1d(output_dim))
                if drop_type == "standard":
                    self.dropout_layers.append(nn.Dropout(dropout))
                elif drop_type == "variational":
                    self.dropout_layers.append(
                        VariationalDropout(output_dim, initial_dropout_rate=dropout)
                    )
                else:
                    raise ValueError(f"Unknown dropout type: {drop_type}")

        assert len(self.layers) == n_layers

    def forward(self, h):
        for i in range(self.n_layers - 1):
            layer = self.layers[i]
            h = layer(h)
            if self.batch_norm:
                h = self.batch_norm_layers[i](h)
            h = self.activation(h)
            h = self.dropout_layers[i](h)
        x_recon = torch.sigmoid(self.layers[-1](h))
        return x_recon


class VAE(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        input_dim: int,
        activation: str = "mish",
        n_layers: int = 3,
        geometry: str = "flat",
        dropout: float = 0.3,
        drop_type: str = "variational",
    ):
        super(VAE, self).__init__()
        self.geometry = geometry

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "mish":
            self.activation = nn.Mish()
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "SiLU":
            self.activation = nn.SiLU()

        self.dimension = generate_dimension(
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
            drop_type=drop_type,
        )
        self.decoder = Decoder(
            activation=self.activation,
            n_layers=n_layers,
            dimension=self.dimension,
            dropout=dropout,
            drop_type=drop_type,
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


def create_vae_model(
    dim: np.ndarray,
    latent_dim_factor: float = 0.05,
    activation: str = "mish",
    n_layers: int = 3,
    geometry: str = "flat",
    dropout: float = 0.3,
) -> VAE:
    hidden_dim = int(np.prod(dim))
    latent_dim = int(np.prod(dim) * latent_dim_factor)

    model = VAE(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        input_dim=hidden_dim,
        activation=activation,
        n_layers=n_layers,
        geometry=geometry,
        dropout=dropout,
    )

    return model
