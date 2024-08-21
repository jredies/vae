import time
import os
import pathlib
import itertools
import sys
import logging
from multiprocessing import Pool

import torch

from vae.models.simple_vae import create_vae_model
from vae.models.training import train_vae, get_loaders
from vae.utils import exception_hook, model_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def run_experiment(
    path: str,
):
    n_layers = 3
    geometry = "flat"
    latent_dim_factor = 0.2
    train_loader, validation_loader, test_loader, dim = get_loaders()

    vae = create_vae_model(
        dim=dim,
        n_layers=n_layers,
        geometry=geometry,
        latent_dim_factor=latent_dim_factor,
    )
    model_name = f"autoencoder"

    train_vae(
        vae=vae,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        dim=dim,
        model_path=path,
        file_name=model_name,
        loss_type="reconstruction",
        gamma=0.25,
        plateau_patience=7,
        patience=15,
        epochs=400,
        scheduler_type="plateau",
        annealing_type="step",
    )

    model_save_path = path / (model_name + ".pth")
    torch.save(vae.state_dict(), model_save_path)


def google_stuff() -> pathlib.Path:
    try:
        from google.colab import drive

        log.info("Running on Google Colab.")
        save_path = "/content/drive/My Drive/thesis/data/"
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        return pathlib.Path(save_path)

    except ImportError:
        log.info("Not running on Google Colab.")
        path = "/Users/joachim/Library/Mobile Documents/com~apple~CloudDocs/thesis/data"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        return pathlib.Path(path)


def main():
    path = google_stuff()

    run_experiment(path=path)


if __name__ == "__main__":
    main()
