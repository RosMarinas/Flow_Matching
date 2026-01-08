"""
CIFAR-10 Training Script with Class Labels (Phase 9)

Trains a class-labeled Flow Matching model on CIFAR-10.
This model can generate samples from specific classes using cross-attention.
"""

import sys
from pathlib import Path
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import time

from src.cfm import FlowMatchingWithLabels
from src.models import UNetWithClassLabels
from src.data import get_cifar10_dataloader
from src.visualize import save_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Train Class-Labeled Flow Matching on CIFAR-10")

    # Model hyperparameters
    parser.add_argument("--model_channels", type=int, default=32, help="Base channels for U-Net")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="Number of res blocks per level")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training hyperparameters
    parser.add_argument("--path", type=str, default="OT", choices=["OT", "VP"], help="Probability path type")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")

    # Checkpointing and Logging
    parser.add_argument("--log_interval", type=int, default=100, help="Batches between logs")
    parser.add_argument("--save_interval", type=int, default=10, help="Epochs between saving checkpoints")
    parser.add_argument("--sample_interval", type=int, default=10, help="Epochs between generating samples")
    parser.add_argument("--output_dir", type=str, default="results/cifar10/with_labels", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


@torch.no_grad()
def sample_with_labels(model, class_labels, num_steps=50, device='cuda'):
    """
    Generate samples from specific class labels.

    Args:
        model: Trained class-labeled model
        class_labels: Class labels tensor, shape (B,)
        num_steps: Number of ODE integration steps
        device: torch device

    Returns:
        samples: Generated images, shape (B, 3, 32, 32)
    """
    model.eval()
    num_samples = class_labels.shape[0]

    # Initialize from noise
    x = torch.randn(num_samples, 3, 32, 32, device=device)
    dt = 1.0 / num_steps

    # Integrate ODE with class conditioning
    for i in range(num_steps):
        t = torch.ones(num_samples, device=device) * (i / num_steps)
        t = t.view(-1, 1, 1, 1)

        # Predict velocity with class conditioning
        v = model(x, t, class_labels)

        # Euler step
        x = x + dt * v

    return x


def train(args):
    # Setup output directories
    output_dir = Path(args.output_dir) / args.path
    ckpt_dir = output_dir / "checkpoints"
    samples_dir = output_dir / "samples"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Training Class-Labeled Flow Matching ({args.path}) on CIFAR-10")
    print(f"Device: {args.device}")
    print(f"Model: UNetWithClassLabels (ch={args.model_channels}, blocks={args.num_res_blocks})")
    print("=" * 60)

    # Data
    print("Loading CIFAR-10...")
    train_loader = get_cifar10_dataloader(
        batch_size=args.batch_size,
        train=True,
        download=True,
        num_workers=4
    )

    # Model
    model = UNetWithClassLabels(
        num_classes=10,
        class_emb_dim=256,
        in_channels=3,
        out_channels=3,
        model_channels=args.model_channels,
        num_res_blocks=args.num_res_blocks,
        channel_mult=(1, 2, 2, 2),
        attention_resolutions=(2,),  # At 16x16 resolution
        dropout=args.dropout,
        num_heads=4
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Flow Matching with labels
    cfm = FlowMatchingWithLabels(model, path_type=args.path, sigma_min=1e-4)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train loop
    start_time = time.time()
    losses = []

    # History for logging
    history = {
        "train_loss": []
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, (x, labels) in enumerate(pbar):  # CHANGED: Capture labels
            x = x.to(args.device)  # (B, 3, 32, 32)
            labels = labels.to(args.device)  # CHANGED: Move labels to device

            # Compute loss with labels
            loss = cfm.compute_loss(x, labels)  # CHANGED: Pass labels

            # Update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            # Log
            if batch_idx % args.log_interval == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        history["train_loss"].append(avg_loss)
        print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.6f}")

        # Save checkpoint
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            ckpt_path = ckpt_dir / f"model_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        # Generate class-conditional samples
        if epoch % args.sample_interval == 0 or epoch == args.epochs:
            print(f"Generating class-labeled samples for epoch {epoch}...")
            model.eval()

            all_samples = []
            for class_id in range(10):
                # Generate 8 samples per class
                class_labels = torch.full((8,), class_id, dtype=torch.long, device=args.device)
                samples = sample_with_labels(model, class_labels, num_steps=50, device=args.device)
                all_samples.append(samples)

            # Concatenate all samples: (10, 8, 3, 32, 32) -> (80, 3, 32, 32)
            all_samples = torch.cat(all_samples, dim=0)

            # Save as grid (10 rows, 8 columns)
            save_path = samples_dir / f"samples_epoch_{epoch}.png"
            save_samples(all_samples, str(save_path), nrow=8, normalize=True, value_range=(-1, 1))
            print(f"Saved samples to {save_path}")

        # Save logs
        with open(output_dir / "log.json", "w") as f:
            json.dump(history, f, indent=4)

    total_time = time.time() - start_time
    print(f"Training finished in {total_time/3600:.2f} hours")

    # Save final model separately
    torch.save(model.state_dict(), output_dir / "model_final.pt")
    print(f"Final model saved to {output_dir}/model_final.pt")


if __name__ == "__main__":
    args = parse_args()
    train(args)
