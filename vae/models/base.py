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


def run_experiment(salt_and_pepper_noise: float = 0.0005):
    path = "logs/aug"

    n_layers = 4
    geometry = "flat"
    latent_dim_factor = 0.2

    train_loader, validation_loader, test_loader, dim = get_loaders()
    log.info(f"salt_and_pepper_noise: {salt_and_pepper_noise}")

    vae = create_vae_model(
        dim=dim,
        n_layers=n_layers,
        geometry=geometry,
        dropout=0.0,
        drop_type="standard",
        latent_dim_factor=latent_dim_factor,
    )

    df_stats = train_vae(
        vae=vae,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        dim=dim,
        model_path=path,
        gamma=1.0,
        epochs=150,
        salt_and_pepper_noise=salt_and_pepper_noise,
    )
    df_stats.to_csv(f"{path}/reg/flat_sn_{salt_and_pepper_noise}.csv")


sys.excepthook = exception_hook


def main():

    max_concurrent_processes = 5
    salt_and_pepper_noises = [0.00025, 0.0005, 0.001, 0.002, 0.0]

    params = list(itertools.product(salt_and_pepper_noises))
    args_list = [(salt_and_pepper_noise) for salt_and_pepper_noise in params]

    with Pool(processes=max_concurrent_processes) as pool:
        pool.starmap(run_experiment, args_list)


if __name__ == "__main__":
    main()
