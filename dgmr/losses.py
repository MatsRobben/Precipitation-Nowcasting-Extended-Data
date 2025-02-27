import torch.nn as nn
import torch.nn.functional as F
import torch


class GridCellLoss(nn.Module):
    """
    Grid Cell Regularizer loss from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf

    """

    def __init__(self, precip_weight_cap=24.0):
        """
        Initialize GridCellLoss.

        Args:
            weight_fn: A function to compute weights for the loss.
            ceil: Custom ceiling value for the weight function.
        """
        super().__init__()
        self.precip_weight_cap = precip_weight_cap

    def forward(self, generated_images, targets):
        """
        Calculates the grid cell regularizer value, assumes generated images are the mean predictions from
        6 calls to the generater (Monte Carlo estimation of the expectations for the latent variable)

        Args:
            generated_images: Mean generated images from the generator
            targets: Ground truth future frames

        Returns:
            Grid Cell Regularizer term
        """
        weights = torch.clip(targets + 1, 0.0, 24.0)
        loss = torch.mean(torch.abs((generated_images - targets) * weights))
        return loss


def loss_grid_regularizer(generated_images, target_images):
    """Grid Cell Regularizer loss from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf"""
    weights = torch.clip(target_images + 1, 0.0, 24.0)
    loss = torch.mean(torch.abs((generated_images - target_images) * weights))
    return loss


def loss_hinge_disc(score_generated, score_real):
    """Discriminator hinge loss."""
    l1 = F.relu(1.0 - score_real)
    l2 = F.relu(1.0 + score_generated)
    loss = torch.mean(l1 + l2)
    return loss


def loss_hinge_gen(score_generated):
    """Generator hinge loss."""
    loss = torch.mean(score_generated)
    return loss
