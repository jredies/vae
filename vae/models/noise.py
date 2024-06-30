import torch


def add_gaussian_noise(inputs, noise_factor=0.5):
    """
    Adds Gaussian noise to the given inputs.

    Args:
        inputs (torch.Tensor): The input tensor to add noise to.
        noise_factor (float, optional): The factor controlling the amount of noise to add. Defaults to 0.5.

    Returns:
        torch.Tensor: The noisy inputs with added Gaussian noise.
    """
    noise = torch.randn_like(inputs) * noise_factor
    noisy_inputs = inputs + noise
    noisy_inputs = torch.clip(noisy_inputs, 0.0, 1.0)
    return noisy_inputs


def add_salt_and_pepper_noise(inputs, noise_factor=0.1):
    """
    Adds salt and pepper noise to the input tensor.

    Args:
        inputs (torch.Tensor): The input tensor to add noise to.
        noise_factor (float, optional): The factor controlling the amount of noise to add. Defaults to 0.1.

    Returns:
        torch.Tensor: The noisy input tensor.
    """
    noisy_inputs = inputs.clone()
    random_matrix = torch.rand_like(inputs)
    noisy_inputs[random_matrix < (noise_factor / 2)] = 0.0
    noisy_inputs[random_matrix > 1 - (noise_factor / 2)] = 1.0
    return noisy_inputs
