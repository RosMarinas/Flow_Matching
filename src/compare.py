"""
Improved Comparison Script for OT vs VP Paths

Improvements:
1. Progress logging during training
2. Better hyperparameters (5000 epochs)
3. Learning rate scheduling
4. More detailed output
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
from visualize import compare_paths, plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Compare OT and VP paths")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="results/toy/checkpoints")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def train_model(model, cfm, dataloader, optimizer, scheduler, device, epochs, path_name, log_interval=100):
    """Train a model with detailed logging."""
    losses = []

    print(f"\nTraining {path_name} model...")
    print("-" * 60)

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

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Print progress
        if epoch % log_interval == 0 or epoch == epochs - 1:
            curr_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:5d}: {path_name} loss = {avg_loss:.6f}, lr = {curr_lr:.6f}")

    return losses


def main():
    args = parse_args()

    print("=" * 60)
    print("Comparing OT and VP Paths on 2D Toy Data (Improved)")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloader
    print("\nLoading dataset...")
    dataloader = get_toy_dataloader(
        dataset="checkerboard",
        n_samples=100000,
        batch_size=args.batch_size,
        shuffle=True,
        device=args.device,
        grid_size=4,
    )
    print(f"✅ Dataset: {len(dataloader)} batches per epoch")

    # Train OT model
    print("\n" + "=" * 60)
    print("Training OT Path Model")
    print("=" * 60)

    model_ot = MLPVectorField(hidden_dim=512, num_layers=5).to(args.device)
    cfm_ot = ConditionalFlowMatching(model_ot, path_type="OT", sigma_min=1e-4)
    optimizer_ot = torch.optim.Adam(model_ot.parameters(), lr=args.lr)
    scheduler_ot = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_ot, T_max=args.epochs, eta_min=1e-5
    )

    losses_ot = train_model(
        model_ot, cfm_ot, dataloader, optimizer_ot, scheduler_ot,
        args.device, args.epochs, "OT", log_interval=200
    )

    torch.save(model_ot.state_dict(), output_dir / "model_ot.pt")
    print(f"✅ OT model saved")
    print(f"Final OT loss: {losses_ot[-1]:.6f}")
    print(f"Best OT loss: {min(losses_ot):.6f}")

    # Train VP model
    print("\n" + "=" * 60)
    print("Training VP Path Model")
    print("=" * 60)

    model_vp = MLPVectorField(hidden_dim=512, num_layers=5).to(args.device)
    cfm_vp = ConditionalFlowMatching(model_vp, path_type="VP", sigma_min=1e-4)
    optimizer_vp = torch.optim.Adam(model_vp.parameters(), lr=args.lr)
    scheduler_vp = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_vp, T_max=args.epochs, eta_min=1e-5
    )

    losses_vp = train_model(
        model_vp, cfm_vp, dataloader, optimizer_vp, scheduler_vp,
        args.device, args.epochs, "VP", log_interval=200
    )

    torch.save(model_vp.state_dict(), output_dir / "model_vp.pt")
    print(f"✅ VP model saved")
    print(f"Final VP loss: {losses_vp[-1]:.6f}")
    print(f"Best VP loss: {min(losses_vp):.6f}")

    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating Comparison Visualizations")
    print("=" * 60)

    fig_dir = Path("report/figures/toy")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Plot training curves
    plot_training_curves(
        losses_ot,
        losses_vp,
        save_path=str(fig_dir / "training_curves_comparison.png"),
    )
    plt.close("all")
    print("✅ Training curves saved")

    # Generate path comparison (KEY FIGURE!)
    model_ot.eval()
    model_vp.eval()

    compare_paths(
        model_ot,
        model_vp,
        num_steps=100,  # More steps for smoother trajectories
        save_path=str(fig_dir / "path_comparison_ot_vs_vp.png"),
    )
    plt.close("all")
    print("✅ Path comparison saved (KEY FIGURE!)")

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"OT final loss: {losses_ot[-1]:.6f} (best: {min(losses_ot):.6f})")
    print(f"VP final loss: {losses_vp[-1]:.6f} (best: {min(losses_vp):.6f})")
    print(f"\nAll figures saved to: {fig_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
