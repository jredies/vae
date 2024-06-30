import typing

import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def load_mnist(
    batch_size=128, validation_split=0.2
) -> typing.Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:

    transform_train = transforms.Compose(
        [
            transforms.RandomRotation(5),  # Random rotations
            # transforms.RandomAffine(
            #     degrees=0, translate=(0.1, 0.1)
            # ),  # Random translations
            # transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),  # Random scaling
            transforms.ToTensor(),
        ]
    )
    transform_test = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        transform=transform_train,
        download=True,
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        transform=transform_test,
        download=True,
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

    dimensionality = np.array(train_dataset.dataset.data[0].shape)

    return train_loader, validation_loader, test_loader, dimensionality
