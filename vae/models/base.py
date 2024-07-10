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
    path = "logs/aug"

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

    df_stats = train_vae(
        vae=vae,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        dim=dim,
        model_path=path,
        gamma=0.1,
        epochs=300,
        loss_type="iwae" if iw_samples > 0 else "standard",
        iw_samples=iw_samples,
    )
    path = "outputs/reg/output"
    _path = pathlib.Path(path)
    _path.mkdir(parents=True, exist_ok=True)

    df_stats.to_csv(_path / f"iw_base_{iw_samples}.csv")


sys.excepthook = exception_hook


def main():
    for x in reversed([0, 3, 10, 30]):
        run_experiment(iw_samples=x)


if __name__ == "__main__":
    main()
