"""
2D Toy Datasets for Flow Matching Experiments

Implements various 2D distributions for visualizing and validating
Flow Matching algorithms:
- Checkerboard: Classic 2D distribution with multiple modes
- Gaussian Mixture: Simple multi-modal distribution
- Spiral: Challenging distribution with continuous structure
"""

import torch
import numpy as np
from typing import Tuple, Literal, Optional
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_cifar10_dataloader(
    batch_size: int = 128,
    train: bool = True,
    download: bool = True,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for the CIFAR-10 dataset.

    Normalizes images to the [-1, 1] range as expected by the
    Flow Matching model.

    Args:
        batch_size: Number of images per batch
        train: Whether to load training or test set
        download: Whether to download the dataset if not present
        num_workers: Number of subprocesses for data loading
        shuffle: Whether to shuffle the data

    Returns:
        dataloader: PyTorch DataLoader
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [0, 1] -> [-1, 1]
        ]
    )

    dataset = datasets.CIFAR10(
        root="./data", train=train, download=download, transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=train,
    )

    return dataloader


def generate_checkerboard(
    n_samples: int = 100000,
    grid_size: int = 4,
    bounds: Tuple[float, float] = (-4, 4),
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate 2D checkerboard distribution.

    Creates alternating 2D squares arranged in a grid pattern.
    This is the classic distribution used in the Flow Matching paper
    (Figure 4) to visualize vector fields and trajectories.

    Args:
        n_samples: Number of samples to generate
        grid_size: Number of squares per row/column (e.g., 4 means 4x4 grid)
        bounds: (min, max) bounds for both x and y dimensions
        device: torch device

    Returns:
        samples: Tensor of shape (n_samples, 2)

    Example:
        >>> samples = generate_checkerboard(1000, grid_size=4)
        >>> print(samples.shape)  # (1000, 2)
    """
    min_val, max_val = bounds
    square_size = (max_val - min_val) / grid_size

    samples = []
    n_per_square = n_samples // (grid_size * grid_size // 2)

    for i in range(grid_size):
        for j in range(grid_size):
            # Checkerboard pattern: alternate between filled and empty
            if (i + j) % 2 == 0:
                # Generate samples in this square
                x_min = min_val + i * square_size
                x_max = x_min + square_size
                y_min = min_val + j * square_size
                y_max = y_min + square_size

                # Uniform distribution within the square
                x = torch.rand(n_per_square, device=device) * (x_max - x_min) + x_min
                y = torch.rand(n_per_square, device=device) * (y_max - y_min) + y_min

                samples.append(torch.stack([x, y], dim=1))

    # Concatenate all squares
    samples = torch.cat(samples, dim=0)

    # If we have extra samples due to rounding, truncate
    if samples.shape[0] > n_samples:
        samples = samples[:n_samples]
    # If we have too few, add more from random squares
    elif samples.shape[0] < n_samples:
        n_extra = n_samples - samples.shape[0]
        extra = generate_checkerboard(
            n_extra, grid_size=grid_size, bounds=bounds, device=device
        )
        samples = torch.cat([samples, extra], dim=0)

    return samples


def generate_gaussian_mixture(
    n_samples: int = 100000,
    n_components: int = 8,
    bounds: Tuple[float, float] = (-4, 4),
    device: str = "cpu",
    random_state: int = 42,
) -> torch.Tensor:
    """
    Generate 2D Gaussian mixture model.

    Creates multiple Gaussian blobs arranged in a circle or grid.

    Args:
        n_samples: Number of samples to generate
        n_components: Number of Gaussian components
        bounds: (min, max) bounds for placing component centers
        device: torch device
        random_state: Random seed for reproducibility

    Returns:
        samples: Tensor of shape (n_samples, 2)
    """
    torch.manual_seed(random_state)

    # Generate component centers in a circle
    angles = torch.linspace(0, 2 * math.pi, n_components + 1)[:n_components]
    radius = (bounds[1] - bounds[0]) / 4
    center_x = torch.zeros(n_components, device=device)
    center_y = torch.zeros(n_components, device=device)

    for i, angle in enumerate(angles):
        center_x[i] = radius * math.cos(angle)
        center_y[i] = radius * math.sin(angle)

    # Generate samples
    samples = []
    n_per_component = n_samples // n_components

    for i in range(n_components):
        # Gaussian around component center
        mean = torch.tensor([center_x[i], center_y[i]], device=device)
        std = 0.3

        component_samples = torch.randn(n_per_component, 2, device=device) * std + mean
        samples.append(component_samples)

    samples = torch.cat(samples, dim=0)

    # Adjust for rounding
    if samples.shape[0] > n_samples:
        samples = samples[:n_samples]
    elif samples.shape[0] < n_samples:
        n_extra = n_samples - samples.shape[0]
        extra = torch.randn(n_extra, 2, device=device) * 0.3
        samples = torch.cat([samples, extra], dim=0)

    return samples


def generate_spiral(
    n_samples: int = 100000,
    n_arms: int = 3,
    noise: float = 0.1,
    bounds: Tuple[float, float] = (-4, 4),
    device: str = "cpu",
    random_state: int = 42,
) -> torch.Tensor:
    """
    Generate 2D spiral distribution.

    Creates a spiral pattern, useful for testing continuous structure learning.

    Args:
        n_samples: Number of samples to generate
        n_arms: Number of spiral arms
        noise: Amount of Gaussian noise to add
        bounds: (min, max) bounds (used for scaling)
        device: torch device
        random_state: Random seed

    Returns:
        samples: Tensor of shape (n_samples, 2)
    """
    torch.manual_seed(random_state)

    samples = []
    n_per_arm = n_samples // n_arms

    for i in range(n_arms):
        # Angle offset for this arm
        angle_offset = 2 * math.pi * i / n_arms

        # Generate spiral points
        t = torch.linspace(0, 1, n_per_arm, device=device)

        # Spiral equation
        radius = 3 * t
        angle = 4 * math.pi * t + angle_offset

        x = radius * torch.cos(angle)
        y = radius * torch.sin(angle)

        # Add noise
        x = x + torch.randn_like(x) * noise
        y = y + torch.randn_like(y) * noise

        samples.append(torch.stack([x, y], dim=1))

    samples = torch.cat(samples, dim=0)

    if samples.shape[0] > n_samples:
        samples = samples[:n_samples]

    return samples


def get_toy_dataloader(
    dataset: Literal["checkerboard", "gaussian_mixture", "spiral"] = "checkerboard",
    n_samples: int = 100000,
    batch_size: int = 256,
    shuffle: bool = True,
    device: str = "cpu",
    **kwargs,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for 2D toy datasets.

    Args:
        dataset: Type of dataset
        n_samples: Total number of samples
        batch_size: Batch size
        shuffle: Whether to shuffle samples
        device: torch device
        **kwargs: Additional arguments for dataset generation

    Returns:
        dataloader: PyTorch DataLoader

    Example:
        >>> dataloader = get_toy_dataloader("checkerboard", batch_size=256)
        >>> for batch in dataloader:
        ...     print(batch.shape)  # (256, 2)
    """
    # Generate samples
    if dataset == "checkerboard":
        samples = generate_checkerboard(n_samples, device=device, **kwargs)
    elif dataset == "gaussian_mixture":
        samples = generate_gaussian_mixture(n_samples, device=device, **kwargs)
    elif dataset == "spiral":
        samples = generate_spiral(n_samples, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Create dataset
    toy_dataset = torch.utils.data.TensorDataset(samples)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        toy_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to >0 for multiprocessing
        drop_last=shuffle,
    )

    return dataloader


class Toy2DDataset(torch.utils.data.Dataset):
    """
    Generic 2D toy dataset class.

    Allows for easy sampling and regeneration.
    """

    def __init__(
        self,
        dataset_type: Literal["checkerboard", "gaussian_mixture", "spiral"] = "checkerboard",
        n_samples: int = 100000,
        device: str = "cpu",
        **kwargs,
    ):
        self.dataset_type = dataset_type
        self.n_samples = n_samples
        self.device = device
        self.kwargs = kwargs

        # Generate samples
        self.samples = self._generate_samples()

    def _generate_samples(self) -> torch.Tensor:
        """Generate samples based on dataset type."""
        if self.dataset_type == "checkerboard":
            return generate_checkerboard(
                self.n_samples, device=self.device, **self.kwargs
            )
        elif self.dataset_type == "gaussian_mixture":
            return generate_gaussian_mixture(
                self.n_samples, device=self.device, **self.kwargs
            )
        elif self.dataset_type == "spiral":
            return generate_spiral(
                self.n_samples, device=self.device, **self.kwargs
            )
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

    def __len__(self) -> int:
        return self.samples.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]

    def resample(self):
        """Regenerate the dataset with new random samples."""
        self.samples = self._generate_samples()


if __name__ == "__main__":
    """
    Test and visualize the 2D toy datasets.
    """
    import matplotlib.pyplot as plt

    print("Generating 2D toy datasets...")

    # Generate checkerboard
    checkerboard = generate_checkerboard(10000, grid_size=4)
    print(f"Checkerboard: {checkerboard.shape}")

    # Generate gaussian mixture
    gaussian = generate_gaussian_mixture(10000, n_components=8)
    print(f"Gaussian Mixture: {gaussian.shape}")

    # Generate spiral
    spiral = generate_spiral(10000, n_arms=3)
    print(f"Spiral: {spiral.shape}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(
        checkerboard[:, 0].cpu().numpy(),
        checkerboard[:, 1].cpu().numpy(),
        s=1,
        alpha=0.5,
    )
    axes[0].set_title("Checkerboard")
    axes[0].set_xlim(-4, 4)
    axes[0].set_ylim(-4, 4)
    axes[0].set_aspect("equal")

    axes[1].scatter(
        gaussian[:, 0].cpu().numpy(), gaussian[:, 1].cpu().numpy(), s=1, alpha=0.5
    )
    axes[1].set_title("Gaussian Mixture")
    axes[1].set_xlim(-4, 4)
    axes[1].set_ylim(-4, 4)
    axes[1].set_aspect("equal")

    axes[2].scatter(
        spiral[:, 0].cpu().numpy(), spiral[:, 1].cpu().numpy(), s=1, alpha=0.5
    )
    axes[2].set_title("Spiral")
    axes[2].set_xlim(-4, 4)
    axes[2].set_ylim(-4, 4)
    axes[2].set_aspect("equal")

    plt.tight_layout()
    plt.savefig("toy_datasets_visualization.png", dpi=150)
    print("✅ Visualization saved to toy_datasets_visualization.png")
