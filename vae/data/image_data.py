import functools
import typing

import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def load_image_data(
    batch_size: int = 128,
    validation_split: float = 0.2,
    dataset_name: str = "mnist",
    rotation: int = 0,
    translate: float = 0.0,
) -> typing.Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:

    train_list = []

    if rotation > 0:
        train_list.append(transforms.RandomRotation(rotation))
    if translate > 0:
        train_list.append(
            transforms.RandomAffine(degrees=0, translate=(translate, translate))
        )
    train_list.append(transforms.ToTensor())

    transform_train = transforms.Compose(train_list)
    transform_test = transforms.Compose([transforms.ToTensor()])

    dataset = getattr(datasets, dataset_name)

    train_dataset = dataset(
        root="./data",
        train=True,
        transform=transform_train,
        download=True,
    )

    test_dataset = dataset(
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


load_mnist = functools.partial(load_image_data, dataset_name="MNIST")
load_emnist = functools.partial(load_image_data, dataset_name="EMNIST")
load_fashion_mnist = functools.partial(load_image_data, dataset_name="FashionMNIST")
