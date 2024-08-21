import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.distributions import normal as dist


def standard_loss(
    x_recon,
    x,
    mu,
    logvar,
    beta: float = 1.0,
    cnn=True,
    **kwargs,
) -> float:
    # Compute binary cross-entropy loss per sample and then take the mean
    bce = nn.functional.binary_cross_entropy(x_recon, x, reduction="none")

    # Sum over all elements per sample
    if not cnn:
        bce = bce.sum(dim=[1])
    else:
        bce = bce.sum(dim=[1, 2, 3])
    bce = bce.mean()  # Mean over the batch

    # Compute KLD loss per sample and then take the mean
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kld = kld.mean()  # Mean over the batch

    # Combine losses to form the ELBO
    elbo = bce + beta * kld
    return elbo


def reconstruction_loss(x_recon, x, **kwargs) -> float:
    ret = F.binary_cross_entropy(x_recon, x, reduction="sum")
    ret = ret / x.size(0)
    return ret


def iwae_loss_fast(model: nn.Module, x, num_samples: int = 5, **kwargs) -> torch.tensor:
    # Get the mean and log variance from the encoder
    mean, logvar = model.encoder(x)
    std = torch.exp(0.5 * logvar)

    # Sample from the latent space
    batch_size, z_dim = mean.size()
    eps = torch.randn(batch_size, num_samples, z_dim, device=mean.device)
    z = mean.unsqueeze(1) + eps * std.unsqueeze(1)

    # Decode the latent samples
    z = z.view(batch_size * num_samples, z_dim)
    recon_x = model.decoder(z)
    recon_x = recon_x.view(batch_size, num_samples, -1)

    # Compute log probabilities
    log_p_x_given_z = -F.binary_cross_entropy(
        recon_x, x.unsqueeze(1).expand_as(recon_x), reduction="none"
    ).sum(dim=2)
    log_p_z = dist.Normal(0, 1).log_prob(z).sum(dim=1).view(batch_size, num_samples)
    log_q_z_given_x = (
        dist.Normal(mean.unsqueeze(1), std.unsqueeze(1))
        .log_prob(z.view(batch_size, num_samples, z_dim))
        .sum(dim=2)
    )

    # Compute log weights
    log_weights = log_p_x_given_z + log_p_z - log_q_z_given_x
    log_sum_exp_weights = torch.logsumexp(log_weights, dim=1)

    # Calculate IWAE loss
    iwae_loss = -torch.mean(
        log_sum_exp_weights
        - torch.log(torch.tensor(num_samples, dtype=torch.float, device=mean.device))
    )

    return iwae_loss


def iwae_loss_fast_cnn(
    model: nn.Module, x, num_samples: int = 5, **kwargs
) -> torch.tensor:
    # Flatten x if it is not already flattened
    if x.dim() > 2:
        x = x.view(x.size(0), -1)

    # Get the mean and log variance from the encoder
    mean, logvar = model.encoder(x)
    std = torch.exp(0.5 * logvar)

    # Sample from the latent space
    batch_size, z_dim = mean.size()
    eps = torch.randn(batch_size, num_samples, z_dim, device=mean.device)
    z = mean.unsqueeze(1) + eps * std.unsqueeze(1)

    # Decode the latent samples
    z = z.view(batch_size * num_samples, z_dim)
    recon_x = model.decoder(z)

    # Assuming recon_x has the shape [batch_size * num_samples, flattened_dimension]
    # Reshape recon_x to match the original input dimensions
    recon_x = recon_x.view(batch_size, num_samples, -1)

    # Expand x to match recon_x's dimensions for the computation
    x_expanded = x.unsqueeze(1).expand(batch_size, num_samples, -1)

    # Debugging: print shapes to verify dimensions
    # print(f"Shape of x: {x.shape}")
    # print(f"Shape of x_expanded: {x_expanded.shape}")
    # print(f"Shape of recon_x: {recon_x.shape}")

    # Compute log probabilities
    log_p_x_given_z = -F.binary_cross_entropy(
        recon_x, x_expanded, reduction="none"
    ).sum(
        dim=2
    )  # Summing over the last dimension

    log_p_z = dist.Normal(0, 1).log_prob(z).sum(dim=1).view(batch_size, num_samples)
    log_q_z_given_x = (
        dist.Normal(mean.unsqueeze(1), std.unsqueeze(1))
        .log_prob(z.view(batch_size, num_samples, z_dim))
        .sum(dim=2)
    )

    # Compute log weights
    log_weights = log_p_x_given_z + log_p_z - log_q_z_given_x
    log_sum_exp_weights = torch.logsumexp(log_weights, dim=1)

    # Calculate IWAE loss
    iwae_loss = -torch.mean(
        log_sum_exp_weights
        - torch.log(torch.tensor(num_samples, dtype=torch.float, device=mean.device))
    )

    return iwae_loss
