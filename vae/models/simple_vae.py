import sys
import typing

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
import pandas as pd

from vae.data.image_data import load_mnist, load_emnist, load_fashion_mnist
from vae.utils import save_model, model_path, exception_hook


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc3_logvar = nn.Linear(hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h1 = self.LeakyReLU(self.fc1(x))
        h2 = self.LeakyReLU(self.fc2(h1))
        mu = self.fc3_mu(h2)
        logvar = self.fc3_logvar(h2)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, z):
        h1 = self.LeakyReLU(self.fc1(z))
        h2 = self.LeakyReLU(self.fc2(h1))
        x_recon = torch.sigmoid(self.fc3(h2))
        return x_recon


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


def loss_function(x_recon, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(x_recon, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD + BCE


def mse_loss_function(
    x_recon,
    x,
):
    MSE = F.mse_loss(x_recon, x, reduction="sum")
    return MSE


def run_model(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    learning_rate: float = 1e-3,
    epochs: int = 50,
) -> typing.Tuple[VAE, pd.DataFrame]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    vae = VAE(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    stats = {
        "train_loss": [],
        "validation_loss": [],
        "test_loss": [],
        "train_mse_loss": [],
        "validation_mse_loss": [],
        "test_mse_loss": [],
    }

    vae.train()
    for epoch in range(epochs):

        train_loss, train_mse_loss = 0, 0

        for _, (data, _) in enumerate(train_loader):
            data = data.view(-1, input_dim).to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = vae(data)
            loss = loss_function(x_recon, data, mu, logvar).to(device)
            mloss = mse_loss_function(x_recon, data).to(device)
            loss.backward()
            train_loss += loss.item()
            train_mse_loss += mloss.item()
            optimizer.step()

        vae.eval()
        validation_loss, validation_mse_loss = 0, 0
        with torch.no_grad():
            for data, _ in validation_loader:
                data = data.view(-1, input_dim).to(device)
                x_recon, mu, logvar = vae(data)
                loss = loss_function(x_recon, data, mu, logvar).to(device)
                mloss = mse_loss_function(x_recon, data).to(device)
                validation_loss += loss.item()
                validation_mse_loss += mloss.item()

        test_loss, test_mse_loss = 0, 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.view(-1, input_dim).to(device)
                x_recon, mu, logvar = vae(data)
                loss = loss_function(x_recon, data, mu, logvar).to(device)
                mloss = mse_loss_function(x_recon, data).to(device)
                test_loss += loss.item()
                test_mse_loss += mloss.item()

        stats["train_loss"].append(train_loss / len(train_loader.dataset))
        stats["train_mse_loss"].append(train_mse_loss / len(train_loader.dataset))

        stats["validation_loss"].append(
            validation_loss / len(validation_loader.dataset)
        )
        stats["validation_mse_loss"].append(
            validation_mse_loss / len(validation_loader.dataset)
        )

        stats["test_loss"].append(test_loss / len(test_loader.dataset))
        stats["test_mse_loss"].append(test_mse_loss / len(test_loader.dataset))

        print(
            f"Epoch {epoch} |"
            f"Train Loss {np.round(stats['train_loss'][-1], 2)} | "
            f"Validation Loss {np.round(stats['validation_loss'][-1], 2)}"
        )

    stats = pd.DataFrame(stats)

    return vae, stats


sys.excepthook = exception_hook


def main():
    learning_rate = 1e-3
    epochs = 50
    batch_size = 128

    train_loader, validation_loader, test_loader, dim = load_fashion_mnist(
        batch_size=batch_size,
    )

    hidden_dim = int(np.prod(dim) * 0.5)
    latent_dim = int(np.prod(dim) * 0.25)

    dim = np.prod(dim)

    vae, stats = run_model(
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        input_dim=dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        learning_rate=learning_rate,
        epochs=epochs,
    )
    save_model(vae, model_path() / "simple_vae.pth")
    stats.to_csv(model_path() / "simple_vae_stats.csv", index=False)


if __name__ == "__main__":
    main()
