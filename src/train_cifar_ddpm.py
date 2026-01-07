"""
CIFAR-10 Training Script for DDPM (Phase 8)

Trains a U-Net based Denoising Diffusion Probabilistic Model (DDPM) on CIFAR-10.
Uses the same architecture as Flow Matching for fair comparison.
"""

import sys
from pathlib import Path
import json
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

from src.ddpm import DDPM
from src.ddpm_solver import ddim_sample
from src.models import UNet
from src.data import get_cifar10_dataloader
from src.visualize import save_samples, plot_cifar_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train DDPM on CIFAR-10")

    # Model hyperparameters
    parser.add_argument("--model_channels", type=int, default=32, help="Base channels for U-Net (must match CFM: 32)")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="Number of res blocks per level")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # DDPM hyperparameters
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "cosine"],
                        help="Beta schedule for DDPM")
    parser.add_argument("--beta_start", type=float, default=1e-4, help="Starting beta value")
    parser.add_argument("--beta_end", type=float, default=2e-2, help="Ending beta value")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")

    # Checkpointing and Logging
    parser.add_argument("--log_interval", type=int, default=100, help="Batches between logs")
    parser.add_argument("--save_interval", type=int, default=100, help="Epochs between saving checkpoints")
    parser.add_argument("--sample_interval", type=int, default=100, help="Epochs between generating samples")
    parser.add_argument("--output_dir", type=str, default="results/cifar10/DDPM", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def train(args):
    # Setup output directories
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    samples_dir = output_dir / "samples"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Training DDPM on CIFAR-10")
    print(f"Device: {args.device}")
    print(f"Model: U-Net (ch={args.model_channels}, blocks={args.num_res_blocks})")
    print(f"Timesteps: {args.num_timesteps}, Beta schedule: {args.beta_schedule}")
    print("=" * 60)

    # Data
    print("Loading CIFAR-10...")
    train_loader = get_cifar10_dataloader(
        batch_size=args.batch_size,
        train=True,
        download=True,
        num_workers=4
    )

    # Model - CRITICAL: use_discrete_time=True for DDPM!
    print(f"\nCreating U-Net model...")
    model = UNet(
        in_channels=3,
        out_channels=3,
        model_channels=args.model_channels,
        num_res_blocks=args.num_res_blocks,
        channel_mult=(1, 2, 2, 2),
        attention_resolutions=(2,),
        dropout=args.dropout,
        num_heads=4,
        use_discrete_time=True,  # Critical for DDPM!
        max_timesteps=args.num_timesteps
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # DDPM
    ddpm = DDPM(
        model,
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end
    )

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # Train loop
    start_time = time.time()
    losses = []

    # History for logging
    history = {
        "train_loss": [],
        "lr": []
    }

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, (x, _) in enumerate(pbar):
            x = x.to(args.device)  # (B, 3, 32, 32), range [-1, 1]

            # Compute DDPM loss
            loss = ddpm.compute_loss(x)

            # Update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            # Log
            if batch_idx % args.log_interval == 0:
                curr_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{curr_lr:.2e}"})

        # Step learning rate scheduler
        scheduler.step()

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        history["train_loss"].append(avg_loss)
        history["lr"].append(optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch:4d}: loss = {avg_loss:.6f}, lr = {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            ckpt_path = ckpt_dir / f"model_epoch{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        # Generate samples
        if epoch % args.sample_interval == 0 or epoch == args.epochs:
            print(f"  Generating samples (NFE=100)...")
            model.eval()
            with torch.no_grad():
                # Generate 64 samples using DDIM
                samples = ddim_sample(
                    model,
                    num_samples=64,
                    input_shape=(3, 32, 32),
                    num_steps=100,
                    eta=0.0,
                    device=args.device,
                    num_timesteps=args.num_timesteps
                )

                # Save samples
                save_path = samples_dir / f"samples_epoch{epoch}.png"
                save_samples(samples, str(save_path), nrow=8, normalize=True, value_range=(-1, 1))
                print(f"  Samples saved: {save_path}")

        # Save logs and plot curves
        with open(output_dir / f"log_{epoch}.json", "w") as f:
            json.dump(history, f, indent=4)

        # Plot training curve
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(losses, linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("DDPM Training Loss", fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "training_metrics.png", dpi=150)
        plt.close()

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time/3600:.2f} hours")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Best loss: {min(losses):.6f}")

    # Save final model
    torch.save(model.state_dict(), output_dir / f"model_final_{args.epochs}epoch.pt")
    print(f"\n✅ Final model saved: {output_dir}/model_final_{args.epochs}epoch.pt")

    # Save final training log
    with open(output_dir / f"log_{args.epochs}.json", "w") as f:
        json.dump(history, f, indent=4)
    print(f"✅ Training log saved: {output_dir}/log_{args.epochs}.json")


if __name__ == "__main__":
    args = parse_args()
    train(args)
