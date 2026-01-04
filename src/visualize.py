"""
Visualization Functions for 2D Flow Matching Experiments

Implements all the visualizations needed for the Project 3 report:
1. Vector field visualization (quiver plots)
2. Generation trajectories
3. OT vs VP path comparison
4. Probability density evolution

These visualizations will produce the key figures for the report,
similar to Figure 4 in the Flow Matching paper.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple, Optional
from pathlib import Path


def plot_vector_field(
    model,
    t_values: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    bounds: Tuple[float, float] = (-4, 4),
    n_grid: int = 20,
    save_path: Optional[str] = None,
    title_prefix: str = "",
) -> plt.Figure:
    """
    Visualize the learned vector field at different time steps.

    Creates a quiver plot showing the direction and magnitude of
    the predicted vector field v_t(x) at different times.

    Args:
        model: Trained vector field network
        t_values: List of time points to visualize
        bounds: (min, max) bounds for the plot
        n_grid: Number of grid points per dimension
        save_path: If provided, save figure to this path
        title_prefix: Prefix for plot titles

    Returns:
        fig: matplotlib Figure object
    """
    fig, axes = plt.subplots(1, len(t_values), figsize=(4 * len(t_values), 4))

    if len(t_values) == 1:
        axes = [axes]

    # Create grid
    x = np.linspace(bounds[0], bounds[1], n_grid)
    y = np.linspace(bounds[0], bounds[1], n_grid)
    X, Y = np.meshgrid(x, y)

    # Flatten grid for network input
    xy_flat = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32
    )
    
    # Get device from model
    device = next(model.parameters()).device
    xy_flat = xy_flat.to(device)

    for ax, t in zip(axes, t_values):
        # Create time tensor
        t_tensor = torch.ones((xy_flat.shape[0], 1), device=device) * t

        # Predict vector field
        model.eval()
        with torch.no_grad():
            v = model(xy_flat, t_tensor).detach().cpu().numpy()

        # Reshape to grid
        U = v[:, 0].reshape(X.shape)
        V = v[:, 1].reshape(Y.shape)

        # Compute magnitude for color
        magnitude = np.sqrt(U**2 + V**2)

        # Plot quiver
        q = ax.quiver(
            X,
            Y,
            U,
            V,
            magnitude,
            cmap="viridis",
            pivot="mid",
            scale=None,
            scale_units="xy",
        )

        ax.set_title(f"{title_prefix}Vector Field at t={t:.2f}")
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        plt.colorbar(q, ax=ax, label="Velocity magnitude")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Vector field visualization saved to {save_path}")

    return fig


def plot_trajectories(
    model,
    solver,
    n_samples: int = 20,
    bounds: Tuple[float, float] = (-4, 4),
    num_steps: int = 100,
    save_path: Optional[str] = None,
    title: str = "Generation Trajectories",
    seed: int = 42,
) -> plt.Figure:
    """
    Visualize generation trajectories from noise to data.

    Shows how samples evolve from N(0,I) to the data distribution
    by integrating the learned ODE.

    Args:
        model: Trained vector field network
        solver: ODE solver function (e.g., euler_solver)
        n_samples: Number of trajectories to visualize
        bounds: (min, max) bounds for the plot
        num_steps: Number of integration steps
        save_path: If provided, save figure to this path
        title: Plot title
        seed: Random seed

    Returns:
        fig: matplotlib Figure object
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get device from model
    device = next(model.parameters()).device

    # Sample initial noise
    x0 = torch.randn(n_samples, 2, device=device)

    # Integrate ODE and record trajectory (no grad needed for visualization)
    with torch.no_grad():
        trajectory = solver(model, x0, num_steps=num_steps, return_trajectory=True)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot each trajectory
    colors = plt.cm.tab10(np.linspace(0, 1, n_samples))

    for i in range(n_samples):
        traj = trajectory[:, i, :].detach().cpu().numpy()  # (num_steps+1, 2)

        # Plot trajectory with color gradient
        points = np.array([traj[:-1], traj[1:]]).transpose(1, 0, 2)
        for j in range(len(points) - 1):
            ax.plot(
                points[j : j + 2, 0],
                points[j : j + 2, 1],
                color=colors[i],
                alpha=0.3,
                linewidth=1,
            )

        # Mark start and end
        ax.scatter(traj[0, 0], traj[0, 1], color=colors[i], marker="o", s=50, label="Start" if i == 0 else "")
        ax.scatter(
            traj[-1, 0],
            traj[-1, 1],
            color=colors[i],
            marker="*",
            s=100,
            label="End" if i == 0 else "",
        )

    ax.set_title(title)
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Trajectory visualization saved to {save_path}")

    return fig


def compare_paths(
    model_ot,
    model_vp,
    bounds: Tuple[float, float] = (-4, 4),
    num_steps: int = 100,
    n_samples: int = 30,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare OT and VP paths side-by-side.

    This is the key visualization showing that OT paths are straight
    while VP paths are curved (reproducing Figure 4 from the paper).

    Args:
        model_ot: Model trained with OT path
        model_vp: Model trained with VP path
        bounds: (min, max) bounds
        num_steps: Number of integration steps
        n_samples: Number of trajectories to show (default: 30)
        save_path: If provided, save figure to this path

    Returns:
        fig: matplotlib Figure object
    """
    from solver import euler_solver
    from data import generate_checkerboard

    # Get device from model
    device = next(model_ot.parameters()).device

    # Generate background checkerboard data for reference
    background_data = generate_checkerboard(n_samples=5000, grid_size=4, device=device)

    # Sample initial noise
    x0 = torch.randn(n_samples, 2, device=device)

    # Integrate both models (no grad needed for visualization)
    with torch.no_grad():
        traj_ot = euler_solver(model_ot, x0, num_steps=num_steps, return_trajectory=True)
        traj_vp = euler_solver(model_vp, x0, num_steps=num_steps, return_trajectory=True)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, n_samples))

    # OT trajectories
    # Plot background data first
    axes[0].scatter(
        background_data[:, 0].detach().cpu().numpy(),
        background_data[:, 1].detach().cpu().numpy(),
        s=1, c='gray', alpha=0.2, label='Target distribution'
    )

    for i in range(n_samples):
        traj = traj_ot[:, i, :].detach().cpu().numpy()
        axes[0].plot(traj[:, 0], traj[:, 1], color=colors[i], alpha=0.6, linewidth=2)
        axes[0].scatter(traj[0, 0], traj[0, 1], color=colors[i], marker="o", s=50)
        axes[0].scatter(traj[-1, 0], traj[-1, 1], color=colors[i], marker="*", s=100)

    axes[0].set_title("OT Path (Straight Lines)", fontsize=14, fontweight="bold")
    axes[0].set_xlim(bounds)
    axes[0].set_ylim(bounds)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')

    # VP trajectories
    # Plot background data first
    axes[1].scatter(
        background_data[:, 0].detach().cpu().numpy(),
        background_data[:, 1].detach().cpu().numpy(),
        s=1, c='gray', alpha=0.2, label='Target distribution'
    )

    for i in range(n_samples):
        traj = traj_vp[:, i, :].detach().cpu().numpy()
        axes[1].plot(traj[:, 0], traj[:, 1], color=colors[i], alpha=0.6, linewidth=2)
        axes[1].scatter(traj[0, 0], traj[0, 1], color=colors[i], marker="o", s=50)
        axes[1].scatter(traj[-1, 0], traj[-1, 1], color=colors[i], marker="*", s=100)

    axes[1].set_title("VP Path (Curved)", fontsize=14, fontweight="bold")
    axes[1].set_xlim(bounds)
    axes[1].set_ylim(bounds)
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    plt.suptitle("Flow Matching: OT vs Diffusion Paths", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Path comparison saved to {save_path}")

    return fig


def plot_density_evolution(
    model,
    solver,
    bounds: Tuple[float, float] = (-4, 4),
    n_bins: int = 50,
    time_points: List[int] = [0, 25, 50, 75, 100],
    n_samples: int = 10000,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize probability density evolution during generation.

    Shows histograms of samples at different time points to
    demonstrate how the distribution emerges from noise.

    Args:
        model: Trained vector field network
        solver: ODE solver
        bounds: (min, max) bounds
        n_bins: Number of histogram bins
        time_points: Time indices to visualize
        n_samples: Number of samples for density estimation
        save_path: If provided, save figure to this path

    Returns:
        fig: matplotlib Figure object
    """
    # Get device from model
    device = next(model.parameters()).device

    # Sample initial noise
    x0 = torch.randn(n_samples, 2, device=device)

    # Integrate full trajectory (no grad needed for visualization)
    with torch.no_grad():
        trajectory = solver(model, x0, num_steps=100, return_trajectory=True)

    # Plot
    fig, axes = plt.subplots(1, len(time_points), figsize=(4 * len(time_points), 4))

    if len(time_points) == 1:
        axes = [axes]

    for ax, t_idx in zip(axes, time_points):
        samples = trajectory[t_idx].detach().cpu().numpy()

        # 2D histogram
        h = ax.hist2d(
            samples[:, 0],
            samples[:, 1],
            bins=n_bins,
            range=[bounds, bounds],
            cmap="Blues",
            density=True,
        )

        ax.set_title(f"t={t_idx/100:.2f}")
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        plt.colorbar(h[3], ax=ax, label="Density")

    plt.suptitle("Probability Density Evolution", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Density evolution saved to {save_path}")

    return fig


def plot_training_curves(
    losses_ot: List[float],
    losses_vp: Optional[List[float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training curves.

    Args:
        losses_ot: Training losses for OT path
        losses_vp: Training losses for VP path (optional)
        save_path: If provided, save figure to this path

    Returns:
        fig: matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(losses_ot, label="OT Path", linewidth=2)
    if losses_vp is not None:
        ax.plot(losses_vp, label="VP Path", linewidth=2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Training curves saved to {save_path}")

    return fig


def save_samples(
    samples: torch.Tensor,
    save_path: str,
    nrow: int = 8,
    normalize: bool = True,
    value_range: Tuple[float, float] = (-1, 1),
):
    """
    Save a batch of samples as a grid image.

    Args:
        samples: Tensor of shape (B, C, H, W)
        save_path: Path to save the image
        nrow: Number of images per row
        normalize: Whether to shift the image to the range (0, 1)
        value_range: Range of the input image
    """
    from torchvision.utils import save_image

    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    save_image(
        samples,
        save_path,
        nrow=nrow,
        normalize=normalize,
        value_range=value_range,
    )
    print(f"✅ Samples saved to {save_path}")


def create_report_figures(
    model_ot,
    model_vp,
    solver,
    output_dir: str = "reports/figures/toy",
):
    """
    Generate all figures needed for the report.

    Args:
        model_ot: Model trained with OT path
        model_vp: Model trained with VP path
        solver: ODE solver function
        output_dir: Directory to save figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Generating report figures...")

    # 1. Vector fields
    plot_vector_field(
        model_ot,
        t_values=[0.0, 0.5, 1.0],
        save_path=str(output_path / "vector_field_ot.png"),
        title_prefix="OT Path - ",
    )

    # 2. Trajectories
    plot_trajectories(
        model_ot,
        solver,
        save_path=str(output_path / "trajectories_ot.png"),
        title="OT Path Generation Trajectories",
    )

    # 3. Path comparison (KEY FIGURE!)
    compare_paths(
        model_ot,
        model_vp,
        save_path=str(output_path / "path_comparison_ot_vs_vp.png"),
    )

    # 4. Density evolution
    plot_density_evolution(
        model_ot,
        solver,
        save_path=str(output_path / "density_evolution_ot.png"),
    )

    print(f"\n✅ All figures saved to {output_dir}/")
