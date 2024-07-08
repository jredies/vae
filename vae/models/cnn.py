import itertools
from multiprocessing import Pool
import pathlib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from vae.models.simple_vae import VAE, Encoder, Decoder
from vae.models.training import train_vae, get_loaders


class CNN_Encoder(nn.Module):
    def __init__(self, latent_dim: int, layers: list, i=1):
        super(CNN_Encoder, self).__init__()

        layer1 = layers[0]
        layer2 = layers[1]
        layer3 = layers[2]
        layer4 = layers[3]

        cnn1 = nn.Conv2d(layer1[0], layer1[1], **layer1[2])
        cnn2 = nn.Conv2d(layer2[0], layer2[1], **layer2[2])
        cnn3 = nn.Conv2d(layer3[0], layer3[1], **layer3[2])
        cnn4 = nn.Conv2d(layer4[0], layer4[1], **layer4[2])

        self.convs = nn.Sequential(
            cnn1,
            nn.BatchNorm2d(i * 16),
            nn.SiLU(),
            cnn2,
            nn.BatchNorm2d(i * 32),
            nn.SiLU(),
            cnn3,
            nn.BatchNorm2d(i * 64),
            nn.SiLU(),
            cnn4,
            nn.BatchNorm2d(i * 128),
            nn.SiLU(),
        )

        self.output_convs = self._output_convs()
        self.output_length = np.prod(self.output_convs)

        self.fc_mu = nn.Linear(self.output_length, latent_dim)
        self.fc_logvar = nn.Linear(self.output_length, latent_dim)

    def _output_convs(self):
        print("Encoder")
        x = torch.rand(1, 1, 28, 28)
        print(x.shape)
        for layer in self.convs:
            x = layer(x)
            print(x.shape)
        return x.shape[1:]

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.convs(x)
        x = x.view(-1, self.output_length)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class CNN_Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_convs: tuple, layers: list, i=1):
        super(CNN_Decoder, self).__init__()

        self.output_dim = output_convs
        self.output_length = np.prod(output_convs)

        self.fc = nn.Linear(
            latent_dim, self.output_length
        )  # Adjusted for 512 latent space

        layer1 = layers[-1]
        layer2 = layers[-2]
        layer3 = layers[-3]
        layer4 = layers[-4]

        self.tcnn1 = nn.ConvTranspose2d(
            layer1[1],
            layer1[0],
            **layer1[2],
            output_padding=1,
        )
        self.tcnn2 = nn.ConvTranspose2d(
            layer2[1],
            layer2[0],
            **layer2[2],
        )
        self.tcnn3 = nn.ConvTranspose2d(
            layer3[1],
            layer3[0],
            **layer3[2],
            output_padding=1,
        )
        self.tcnn4 = nn.ConvTranspose2d(
            layer4[1],
            layer4[0],
            **layer4[2],
            output_padding=1,
        )

        self.dconvs = nn.Sequential(
            self.tcnn1,
            nn.BatchNorm2d(i * 64),
            nn.SiLU(),
            self.tcnn2,
            nn.BatchNorm2d(i * 32),
            nn.SiLU(),
            self.tcnn3,
            nn.BatchNorm2d(i * 16),
            nn.SiLU(),
            self.tcnn4,
        )

        self._output_convs()

    def _output_convs(self):
        print("Decoder")
        x = torch.rand(1, *self.output_dim)
        print(x.shape)
        for layer in self.dconvs:
            x = layer(x)
            print(x.shape)

    def forward(self, h):
        x = self.fc(h)
        x = x.view(-1, *self.output_dim)
        x = self.dconvs(x)
        x = torch.sigmoid(x)
        return x


class CNN_VAE(nn.Module):
    def __init__(self, iw_samples=0, latent_dim=50, i=1):
        super(CNN_VAE, self).__init__()

        layers = [
            (1, i * 16, {"kernel_size": 3, "stride": 2, "padding": 1}),
            (i * 16, i * 32, {"kernel_size": 3, "stride": 2, "padding": 1}),
            (i * 32, i * 64, {"kernel_size": 3, "stride": 2, "padding": 1}),
            (i * 64, i * 128, {"kernel_size": 3, "stride": 2, "padding": 1}),
        ]

        self.encoder = CNN_Encoder(latent_dim=latent_dim, layers=layers, i=i)
        self.output_convs = self.encoder.output_convs
        self.decoder = CNN_Decoder(
            latent_dim=latent_dim,
            output_convs=self.output_convs,
            layers=layers,
            i=i,
        )
        self.iw_samples = iw_samples

    def forward(self, x):
        if self.iw_samples > 0:
            mu, logvar = self.encoder(x)
            z_samples = [
                self.reparameterize(mu, logvar) for _ in range(self.iw_samples)
            ]
            reconstructions = [self.decoder(z) for z in z_samples]
            return reconstructions, mu, logvar
        else:
            mu, logvar = self.encoder(x)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decoder(z)
            return x_recon, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


def run_experiment(i=1, latent_factor=0.2):
    path = "outputs/cnn/output"
    _path = pathlib.Path().mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader, test_loader, dim = get_loaders()

    length = np.prod(dim)
    latent_dim = int(length * latent_factor)
    print("Latent dim: ", latent_dim)

    model = CNN_VAE(latent_dim=latent_dim, i=i)

    df_stats = train_vae(
        vae=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        dim=dim,
        epochs=150,
        gamma=1.0,
        model_path=path,
        cnn=True,
        loss_type="standard",
        iw_samples=0,
    )
    df_stats.to_csv(_path / f"cnn_i_{i}_latent_{latent_factor}.csv")


def main():
    max_concurrent_processes = 4

    latent_dim_factors = [0.025, 0.05, 0.1, 0.2]
    iss = [
        # 1,
        # 2,
        # 3,
        # 4,
        # 5,
        6,
    ]

    params = list(itertools.product(iss, latent_dim_factors))
    args_list = [(i, latent_dim_factor) for i, latent_dim_factor in params]

    with Pool(processes=max_concurrent_processes) as pool:
        pool.starmap(run_experiment, args_list)


if __name__ == "__main__":
    main()
