import pathlib
import itertools
import sys
import logging
from multiprocessing import Pool

from vae.models.simple_vae import create_vae_model
from vae.models.training import train_vae, get_loaders
from vae.utils import exception_hook, model_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def run_experiment(spectral_norm: bool):
    path = "logs/aug"

    n_layers = 4
    geometry = "flat"
    latent_dim_factor = 0.2

    train_loader, validation_loader, test_loader, dim = get_loaders()
    log.info(f"Spec: {spectral_norm}")

    vae = create_vae_model(
        dim=dim,
        n_layers=n_layers,
        geometry=geometry,
        latent_dim_factor=latent_dim_factor,
        spectral_norm=spectral_norm,
    )

    df_stats = train_vae(
        vae=vae,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        dim=dim,
        model_path=path,
        gamma=0.1,
        epochs=300,
        salt_and_pepper_noise=0.0,
    )
    path = "outputs/reg/output"
    _path = pathlib.Path(path)
    _path.mkdir(parents=True, exist_ok=True)

    df_stats.to_csv(f"{_path}/best_model_spec_{spectral_norm}.csv")


sys.excepthook = exception_hook


def main():

    max_concurrent_processes = 6

    specs = [
        True,
        False,
    ]

    params = list(itertools.product(specs))
    args_list = [spec for spec in params]

    with Pool(processes=max_concurrent_processes) as pool:
        pool.starmap(run_experiment, args_list)


if __name__ == "__main__":
    main()
