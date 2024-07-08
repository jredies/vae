from vae.models.training import train_vae, get_loaders

from vae.models.cnn3 import vae


def main():
    path = "logs/cnn"
    train_loader, validation_loader, test_loader, dim = get_loaders()

    # vae = VAE()

    df_stats = train_vae(
        vae=vae,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        dim=dim,
        model_path=path,
        gamma=0.0,
        epochs=100,
        salt_and_pepper_noise=0.0,
        cnn=True,
    )
    df_stats.to_csv(f"{path}/cvae.csv")


if __name__ == "__main__":
    main()
