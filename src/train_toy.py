"""
Improved Training Script for 2D Toy Experiments

Improvements:
1. Progress logging during training
2. Better hyperparameters
3. Learning rate scheduling
4. More training epochs by default
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from cfm import ConditionalFlowMatching
from solver import euler_solver
from models import MLPVectorField
from data import get_toy_dataloader
from visualize import (
    plot_vector_field,
    plot_trajectories,
    compare_paths,
    plot_training_curves,
    plot_flow_evolution,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Flow Matching on 2D toy data")
    parser.add_argument("--path", type=str, default="OT", choices=["OT", "VP"])
    parser.add_argument("--hidden_dim", type=int, default=128, help="Reduced from 512")
    parser.add_argument("--num_layers", type=int, default=3, help="Reduced from 8")
    parser.add_argument("--epochs", type=int, default=2000, help="Reasonable convergence for 2D")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dataset", type=str, default="checkerboard")
    parser.add_argument("--grid_size", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="results/toy/checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def train_model(model, cfm, dataloader, optimizer, scheduler, device, epochs, log_interval):
    """Train a model with logging."""
    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            x1 = batch[0].to(device)
            loss = cfm.compute_loss(x1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        # Step learning rate scheduler
        if scheduler is not None:
            scheduler.step()

        # Print progress
        if epoch % log_interval == 0 or epoch == epochs - 1:
            curr_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:5d}: loss = {avg_loss:.6f}, lr = {curr_lr:.6f}")

    return losses


def main():
    args = parse_args()

    print("=" * 60)
    print(f"Training {args.path} Path Model on 2D Toy Data")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir) / args.path
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloader
    print("\nLoading dataset...")
    dataloader = get_toy_dataloader(
        dataset=args.dataset,
        n_samples=100000,
        batch_size=args.batch_size,
        shuffle=True,
        device=args.device,
        grid_size=args.grid_size,
    )

    # Create model
    print(f"\nCreating model...")
    model = MLPVectorField(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # Create CFM
    cfm = ConditionalFlowMatching(model, path_type=args.path, sigma_min=1e-4)

    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # Train
    print(f"\nTraining {args.path} model...")
    losses = train_model(
        model, cfm, dataloader, optimizer, scheduler,
        args.device, args.epochs, args.log_interval
    )

    # Save model
    torch.save(model.state_dict(), output_dir / "model_final.pt")
    print(f"\n✅ Model saved: {output_dir}/model_final.pt")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Best loss: {min(losses):.6f}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    model.eval()

    fig_dir = Path("report/figures/toy")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Training curve
    plot_training_curves(
        losses,
        save_path=str(fig_dir / f"training_curve_{args.path}.png"),
    )
    plt.close("all")

    # Vector field
    plot_vector_field(
        model,
        t_values=[0.0, 0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        save_path=str(fig_dir / f"vector_field_{args.path}.png"),
        title_prefix=f"{args.path} Path - ",
    )
    plt.close("all")

    # Trajectories
    plot_trajectories(
        model,
        euler_solver,
        n_samples=100,
        num_steps=100,  # More steps for smoother trajectories
        save_path=str(fig_dir / f"trajectories_{args.path}.png"),
        title=f"{args.path} Path - Generation Trajectories",
    )
    plt.close("all")

    # Flow Evolution
    plot_flow_evolution(
        model,
        euler_solver,
        save_path=str(fig_dir / f"flow_evolution_{args.path}.png"),
        title=f"{args.path} Path - Flow Evolution",
    )
    plt.close("all")

    print(f"✅ All visualizations saved to {fig_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
