import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

from vae.data.mnist import load_mnist
from vae.utils import save_model, model_path
import torch.nn.functional as F


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
    # MSE = F.mse_loss(x_recon, x, reduction="sum")

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD + BCE


def main(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
) -> VAE:
    vae = VAE(input_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    vae.train()
    for epoch in range(epochs):
        train_loss = 0
        for _, (data, _) in enumerate(train_loader):
            data = data.view(-1, input_dim)
            optimizer.zero_grad()
            x_recon, mu, logvar = vae(data)
            loss = loss_function(x_recon, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}")

        vae.eval()
        validation_loss = 0
        with torch.no_grad():
            for data, _ in validation_loader:
                data = data.view(-1, input_dim)
                x_recon, mu, logvar = vae(data)
                loss = loss_function(x_recon, data, mu, logvar)
                validation_loss += loss.item()

        avg_validation_loss = validation_loss / len(validation_loader.dataset)
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_validation_loss}")

    return vae


if __name__ == "__main__":
    learning_rate = 1e-3
    epochs = 100
    hidden_dim = 400
    latent_dim = 20
    batch_size = 128

    train_loader, validation_loader, test_loader, dim = load_mnist(
        batch_size=batch_size,
    )

    dim = np.prod(dim)

    vae = main(
        train_loader=train_loader,
        validation_loader=validation_loader,
        input_dim=dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    )
    save_model(vae, model_path() / "simple_vae.pth")
