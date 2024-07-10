import logging
import itertools
import pathlib
from multiprocessing import Pool

import numpy as np

import torch
import torch.nn as nn

from vae.models.training import train_vae, get_loaders


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


class CNN_Encoder(nn.Module):
    def __init__(self, latent_dim: int, layers: list, i=1, spectral_norm=False):
        super(CNN_Encoder, self).__init__()

        layer1 = layers[0]
        layer2 = layers[1]
        layer3 = layers[2]
        layer4 = layers[3]

        cnn1 = nn.Conv2d(layer1[0], layer1[1], **layer1[2])
        cnn2 = nn.Conv2d(layer2[0], layer2[1], **layer2[2])
        cnn3 = nn.Conv2d(layer3[0], layer3[1], **layer3[2])
        cnn4 = nn.Conv2d(layer4[0], layer4[1], **layer4[2])

        if spectral_norm:
            cnn1 = nn.utils.spectral_norm(cnn1)
            cnn2 = nn.utils.spectral_norm(cnn2)
            cnn3 = nn.utils.spectral_norm(cnn3)
            cnn4 = nn.utils.spectral_norm(cnn4)

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
    def __init__(
        self,
        latent_dim: int,
        output_convs: tuple,
        layers: list,
        i=1,
        spectral_norm=False,
    ):
        super(CNN_Decoder, self).__init__()

        self.output_dim = output_convs
        self.output_length = np.prod(output_convs)

        self.fc = nn.Linear(latent_dim, self.output_length)

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

        if spectral_norm:
            self.tcnn1 = nn.utils.spectral_norm(self.tcnn1)
            self.tcnn2 = nn.utils.spectral_norm(self.tcnn2)
            self.tcnn3 = nn.utils.spectral_norm(self.tcnn3)
            self.tcnn4 = nn.utils.spectral_norm(self.tcnn4)

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
    def __init__(self, iw_samples=0, latent_dim=50, i=1, spectral_norm=False):
        super(CNN_VAE, self).__init__()
        self.spectral_norm = spectral_norm

        layers = [
            (1, i * 16, {"kernel_size": 3, "stride": 2, "padding": 1}),
            (i * 16, i * 32, {"kernel_size": 3, "stride": 2, "padding": 1}),
            (i * 32, i * 64, {"kernel_size": 3, "stride": 2, "padding": 1}),
            (i * 64, i * 128, {"kernel_size": 3, "stride": 2, "padding": 1}),
        ]

        self.encoder = CNN_Encoder(
            latent_dim=latent_dim,
            layers=layers,
            i=i,
            spectral_norm=spectral_norm,
        )
        self.output_convs = self.encoder.output_convs
        self.decoder = CNN_Decoder(
            latent_dim=latent_dim,
            output_convs=self.output_convs,
            layers=layers,
            i=i,
            spectral_norm=spectral_norm,
        )
        self.iw_samples = iw_samples

    def forward(self, x, noise_parameter: float = 0.0):
        if self.iw_samples > 0:
            mu, logvar = self.encoder(x)
            z_samples = [
                self.reparameterize(mu, logvar, noise_parameter=noise_parameter)
                for _ in range(self.iw_samples)
            ]
            reconstructions = [self.decoder(z) for z in z_samples]
            return reconstructions, mu, logvar
        else:
            mu, logvar = self.encoder(x)
            z = self.reparameterize(mu, logvar, noise_parameter=noise_parameter)
            x_recon = self.decoder(z)
            return x_recon, mu, logvar

    def reparameterize(self, mu, logvar, noise_parameter: float = 0.0):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        if noise_parameter > 0.0:
            noise = torch.randn_like(z) * noise_parameter
            z += noise

        return z


def run_experiment(
    latent_noise_std: float,
):
    i = 5
    latent_factor = 0.2

    path = "outputs/cnn/output"
    _path = pathlib.Path(path)
    _path.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader, test_loader, dim = get_loaders()

    length = np.prod(dim)
    latent_dim = int(length * latent_factor)

    model = CNN_VAE(latent_dim=latent_dim, i=i, spectral_norm=False)
    log.info(f"Latent noise std: {latent_noise_std}")

    df_stats = train_vae(
        vae=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        dim=dim,
        epochs=300,
        model_path=path,
        cnn=True,
        loss_type="standard",
        iw_samples=0,
        salt_and_pepper_noise=0.0,
        gamma=0.1,
        latent_noise=latent_noise_std,
    )
    df_stats.to_csv(_path / f"best_model_latent_{latent_noise_std}.csv")


def main():
    max_concurrent_processes = 2

    latent_noise_std = [
        0.0,
        # 0.01,
        0.02,
        # 0.03,
        0.04,
        # 0.05,
        0.06,
        # 0.07,
        0.08,
    ]

    params = list(itertools.product(latent_noise_std))
    args_list = [lns for lns in params]

    for latent_noise in latent_noise_std:
        run_experiment(latent_noise_std=latent_noise)

    # with Pool(processes=max_concurrent_processes) as pool:
    # pool.starmap(run_experiment, args_list)


if __name__ == "__main__":
    main()
