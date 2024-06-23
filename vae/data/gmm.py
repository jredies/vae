import typing

import numpy as np

from scipy.stats import multivariate_normal

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


def generate_gmm() -> typing.Tuple[np.ndarray, np.ndarray]:
    k = 4
    dimension = 5

    means = [
        np.array([0, 0, 0, 0, 0]),
        np.array([10, 10, 10, 10, 10]),
        np.array([20, 0, 20, 0, 20]),
        np.array([-10, 10, -10, 0, 0]),
    ]
    covariances = [
        np.eye(dimension),
        0.5 * np.eye(dimension),
        np.eye(dimension),
        0.3 * np.eye(dimension),
    ]
    mixing_weights = [0.25, 0.25, 0.25, 0.25]

    n_samples = 50_000

    samples = np.zeros((n_samples, dimension))
    component_choices = np.random.choice(k, size=n_samples, p=mixing_weights)

    for i in range(k):
        n_i = np.sum(component_choices == i)
        if n_i > 0:
            samples[component_choices == i] = multivariate_normal.rvs(
                mean=means[i], cov=covariances[i], size=n_i
            )

    X = samples
    Y = component_choices

    return X, Y


DIMENSION = 5


def generate_gmm(n_samples: int = 50_000) -> typing.Tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)

    k = 4

    means = [
        np.array([0, 0, 0, 0, 0]),
        np.array([10, 10, 10, 10, 10]),
        np.array([20, 0, 20, 0, 20]),
        np.array([-10, 10, -10, 0, 0]),
    ]
    covariances = [
        np.eye(DIMENSION),
        0.5 * np.eye(DIMENSION),
        np.eye(DIMENSION),
        0.3 * np.eye(DIMENSION),
    ]
    mixing_weights = [0.25, 0.25, 0.25, 0.25]

    samples = np.zeros((n_samples, DIMENSION))
    component_choices = np.random.choice(k, size=n_samples, p=mixing_weights)

    for i in range(k):
        n_i = np.sum(component_choices == i)
        if n_i > 0:
            samples[component_choices == i] = multivariate_normal.rvs(
                mean=means[i], cov=covariances[i], size=n_i
            )

    X = samples
    Y = component_choices

    return X, Y


class GMMDataset(Dataset):
    def __init__(
        self,
        n_samples: int = 50_000,
        train: bool = True,
        transform: typing.Optional[typing.Callable] = None,
    ):
        self.train = train
        self.transform = transform

        X, Y = generate_gmm(n_samples=n_samples)
        n_samples = X.shape[0]

        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).long()

        if self.train:
            self.X = X[: int(0.8 * n_samples)]
            self.Y = Y[: int(0.8 * n_samples)]
        else:
            self.X = X[int(0.8 * n_samples) :]
            self.Y = Y[int(0.8 * n_samples) :]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample_X = self.X[idx]
        sample_Y = self.Y[idx]
        if self.transform:
            sample_X = self.transform(sample_X)
        return sample_X, sample_Y


def min_max_scale(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    return (tensor - min_val) / (max_val - min_val)


def load_gmm(
    batch_size=128, validation_split=0.2
) -> typing.Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:

    train_dataset = GMMDataset(train=True, transform=min_max_scale)

    test_dataset = GMMDataset(
        train=False,
        transform=min_max_scale,
    )

    train_size = int((1 - validation_split) * len(train_dataset))
    validation_size = len(train_dataset) - train_size

    train_dataset, validation_dataset = random_split(
        train_dataset,
        [train_size, validation_size],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    dimensionality = DIMENSION
    return train_loader, validation_loader, test_loader, dimensionality
