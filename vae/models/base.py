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


def run_experiment(iw_samples):
    n_layers = 4
    geometry = "flat"
    latent_dim_factor = 0.2

    train_loader, validation_loader, test_loader, dim = get_loaders()

    vae = create_vae_model(
        dim=dim,
        n_layers=n_layers,
        geometry=geometry,
        latent_dim_factor=latent_dim_factor,
    )
    log.info(f"Running iw_samples: {iw_samples}")

    path = "outputs/reg/output"
    _path = pathlib.Path(path)
    _path.mkdir(parents=True, exist_ok=True)
    file_name = f"big_k_{iw_samples}.csv"

    df_stats = train_vae(
        vae=vae,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        dim=dim,
        model_path=path,
        file_name=file_name,
        loss_type="iwae" if iw_samples > 0 else "standard",
        iw_samples=iw_samples,
        gamma=0.25,
        plateau_patience=7,
        patience=15,
        epochs=300,
        scheduler_type="plateau",
    )


sys.excepthook = exception_hook


def main():
    for x in list(
        [
            0,
            3,
            10,
        ]
    ):
        run_experiment(iw_samples=x)


if __name__ == "__main__":
    main()
