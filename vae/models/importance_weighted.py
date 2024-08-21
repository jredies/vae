import sys

import numpy as np

import torch
import torch.nn.functional as F

from vae.models.simple_vae import VAE
from vae.models.training import get_loaders, train_vae


from vae.utils import exception_hook


class ImportanceWeightedVAE(VAE):

    def __init__(
        self,
        n_samples: int = 5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_samples = n_samples

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z_samples = [self.reparameterize(mu, logvar) for _ in range(self.n_samples)]
        reconstructions = [self.decoder(z) for z in z_samples]
        return reconstructions, mu, logvar


def create_iw_vae_model(
    dim: np.ndarray,
    latent_dim_factor: float = 0.05,
    *args,
    **kwargs,
) -> VAE:
    hidden_dim = int(np.prod(dim))
    latent_dim = int(np.prod(dim) * latent_dim_factor)

    model = ImportanceWeightedVAE(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        input_dim=hidden_dim,
        *args,
        **kwargs,
    )

    return model


def main():
    path = "logs/aug"

    train_loader, validation_loader, test_loader, dim = get_loaders()

    latent_factor = 0.05

    vae = create_iw_vae_model(
        dim=dim,
        latent_dim_factor=latent_factor,
        n_samples=2,
    )

    df_stats = train_vae(
        vae=vae,
        train_loader=train_loader,
        val_loader=validation_loader,
        test_loader=test_loader,
        dim=dim,
        model_path=path,
        loss_function_type="importance_weighted",
        gamma=0.25,
    )


sys.excepthook = exception_hook


if __name__ == "__main__":
    main()
