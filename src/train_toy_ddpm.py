"""
DDPM Training Script for 2D Toy Data

This script trains a Denoising Diffusion Probabilistic Model (DDPM) on 2D toy datasets.
Uses the same architecture as Flow Matching for fair comparison.

Architecture: MLPVectorField(hidden_dim=512, num_layers=5) - matches CFM models
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from src.ddpm import DDPM
from src.ddpm_solver import ddim_sample
from src.models import MLPVectorField
from src.data import get_toy_dataloader
from src.visualize import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train DDPM on 2D toy data")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension (must match CFM: 128)")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers (must match CFM: 3)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dataset", type=str, default="checkerboard")
    parser.add_argument("--grid_size", type=int, default=4)
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="results/toy/DDPM")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def train_model(model, ddpm, dataloader, optimizer, scheduler, device, epochs, log_interval, output_dir):
    """Train a DDPM model with logging and checkpointing."""
    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            x1 = batch[0].to(device)
            loss = ddpm.compute_loss(x1)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

        # Save checkpoint
        if epoch % args.save_interval == 0 and epoch > 0:
            checkpoint_path = output_dir / f"model_epoch{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"    Checkpoint saved: {checkpoint_path}")

    return losses


def main():
    global args
    args = parse_args()

    print("=" * 60)
    print("Training DDPM on 2D Toy Data")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Model: MLPVectorField(hidden_dim={args.hidden_dim}, num_layers={args.num_layers})")
    print(f"Timesteps: {args.num_timesteps}")
    print(f"Beta schedule: {args.beta_schedule}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
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

    # Create model with discrete time embedding
    print(f"\nCreating model...")
    model = MLPVectorField(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_discrete_time=True,  # Critical for DDPM!
        max_timesteps=args.num_timesteps
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # Create DDPM
    ddpm = DDPM(
        model,
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule
    )

    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # Train
    print(f"\nTraining DDPM model...")
    losses = train_model(
        model, ddpm, dataloader, optimizer, scheduler,
        args.device, args.epochs, args.log_interval, output_dir
    )

    # Save final model
    torch.save(model.state_dict(), output_dir / "model_final.pt")
    print(f"\n[OK] Model saved: {output_dir}/model_final.pt")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Best loss: {min(losses):.6f}")

    # Save training curve
    print("\nSaving training curve...")
    fig_dir = Path("report/figures/toy")
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses, linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("DDPM Training Loss", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "ddpm_training_curve.png", dpi=150)
    print(f"[OK] Training curve saved: {fig_dir}/ddpm_training_curve.png")

    # Generate sample visualizations at different NFE values
    print("\nGenerating samples at different NFE values...")
    model.eval()

    nfe_values = [10, 20,30,40,50,60,80, 100]
    fig, axes = plt.subplots(2, 4, figsize=(12, 12))
    axes = axes.flatten()

    with torch.no_grad():
        for idx, nfe in enumerate(nfe_values):
            print(f"  Sampling with NFE={nfe}...")
            samples = ddim_sample(
                model,
                num_samples=1000,
                input_shape=(2,),
                num_steps=nfe,
                device=args.device,
                schedule=args.beta_schedule
            )

            ax = axes[idx]
            ax.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), s=1, alpha=0.5)
            ax.set_title(f"DDPM/DDIM Samples (NFE={nfe})", fontsize=12)
            ax.set_xlabel("x", fontsize=10)
            ax.set_ylabel("y", fontsize=10)
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(fig_dir / "ddpm_samples_nfe_comparison.png", dpi=150)
    print(f"[OK] Samples saved: {fig_dir}/ddpm_samples_nfe_comparison.png")

    # Generate DDPM evolution visualization (analogous to Flow Matching Figure 1)
    print("\nGenerating DDPM evolution visualization...")
    from src.visualize import plot_ddpm_evolution

    plot_ddpm_evolution(
        model,
        bounds=(-4, 4),
        timesteps=[1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 0],
        n_samples=10000,
        eta=0.0,
        device=args.device,
        beta_schedule=args.beta_schedule,
        save_path=fig_dir / "ddpm_evolution.png",
        title="DDPM Reverse Diffusion Process"
    )
    print(f"[OK] Evolution saved: {fig_dir}/ddpm_evolution.png")

    print("\n" + "=" * 60)
    print("[OK] Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
