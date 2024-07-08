import torch

import torch.nn as nn


def main():
    x = torch.rand(2, 1, 28, 28)
    print(x.shape)

    cnn1 = nn.Conv2d(
        1, 16, kernel_size=3, stride=2, padding=1
    )  # [1, 32 , 32] -> [16, 16, 16]

    cnn2 = nn.Conv2d(
        16, 32, kernel_size=3, stride=2, padding=1
    )  # [16, 16, 16] -> [32, 8, 8]

    x = cnn1(x)
    print(x.shape)

    x = cnn2(x)
    print(x.shape)

    output_shape = 32 * 4 * 4
    output_dim = (32, 4, 4)

    fc1 = nn.Linear(output_shape, 64)
    fc2 = nn.Linear(64, output_shape)

    x = x.view(-1, output_shape)
    x = fc1(x)
    print(x.shape)

    x = fc2(x)
    print(x.shape)
    print("....")

    x = x.view(-1, *output_dim)
    print(x.shape)

    tcnn1 = nn.ConvTranspose2d(
        32, 16, kernel_size=3, stride=2, output_padding=1, padding=1
    )
    x = tcnn1(x)
    print(x.shape)

    tcnn3 = nn.ConvTranspose2d(
        16, 1, kernel_size=3, stride=2, output_padding=1, padding=1
    )
    x = tcnn3(x)
    print(x.shape)


if __name__ == "__main__":
    main()
