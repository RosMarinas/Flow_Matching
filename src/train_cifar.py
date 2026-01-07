"""
CIFAR-10 Training Script for Flow Matching (Phase 3)

Trains a U-Net based Conditional Flow Matching model on CIFAR-10.
Supports OT (Optimal Transport) and VP (Variance Preserving) paths.
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

from src.cfm import ConditionalFlowMatching
from src.models import UNet
from src.data import get_cifar10_dataloader
from src.solver import euler_solver
from src.visualize import save_samples, plot_cifar_metrics
from src.metrics import compute_nll


def parse_args():
    parser = argparse.ArgumentParser(description="Train Flow Matching on CIFAR-10")
    
    # Model hyperparameters
    parser.add_argument("--model_channels", type=int, default=128, help="Base channels for U-Net")
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
    parser.add_argument("--sample_interval", type=int, default=5, help="Epochs between generating samples")
    parser.add_argument("--output_dir", type=str, default="results/cifar10", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_nll", action="store_true", help="Compute NLL during training (slow)")
    
    return parser.parse_args()


def train(args):
    # Setup output directories
    output_dir = Path(args.output_dir) / args.path
    ckpt_dir = output_dir / "checkpoints"
    samples_dir = output_dir / "samples"
    
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"Training Flow Matching ({args.path}) on CIFAR-10")
    print(f"Device: {args.device}")
    print(f"Model: U-Net (ch={args.model_channels}, blocks={args.num_res_blocks})")
    print("=" * 60)

    # Data
    print("Loading CIFAR-10...")
    train_loader = get_cifar10_dataloader(
        batch_size=args.batch_size,
        train=True,
        download=True,
        num_workers=4
    )
    
    # Get a fixed batch for evaluation
    eval_batch = next(iter(train_loader))[0].to(args.device)
    # Use smaller batch for NLL to save time
    eval_batch = eval_batch[:16] 

    # Model
    model = UNet(
        in_channels=3,
        out_channels=3,
        model_channels=args.model_channels,
        num_res_blocks=args.num_res_blocks,
        channel_mult=(1, 2, 2, 2),
        attention_resolutions=(2,),  # At 16x16 resolution (32/2)
        dropout=args.dropout,
        num_heads=4
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Flow Matching
    cfm = ConditionalFlowMatching(model, path_type=args.path, sigma_min=1e-4)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Train loop
    start_time = time.time()
    losses = []
    
    # History for logging
    history = {
        "train_loss": [],
        "nll": []  # List of (epoch, value)
    }
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, (x, _) in enumerate(pbar):
            x = x.to(args.device)  # (B, 3, 32, 32)
            
            # Compute loss
            loss = cfm.compute_loss(x)
            
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

        # Generate samples and Evaluate
        if epoch % args.sample_interval == 0 or epoch == args.epochs:
            print("Generating samples...")
            model.eval()
            with torch.no_grad():
                # Sample 64 images
                x0 = torch.randn(64, 3, 32, 32, device=args.device)
                samples = euler_solver(model, x0, num_steps=50)
                
                # Save samples
                save_path = samples_dir / f"samples_epoch_{epoch}.png"
                save_samples(samples, str(save_path), nrow=8, normalize=True, value_range=(-1, 1))
                
            if args.eval_nll:
                print("Computing NLL (this may take a while)...")
                try:
                    # Using eval_batch defined earlier
                    nll, std = compute_nll(model, eval_batch, method="dopri5")
                    print(f"NLL: {nll:.4f} +/- {std:.4f} bits/dim")
                    history["nll"].append((epoch, nll))
                except Exception as e:
                    print(f"NLL computation failed: {e}")

        # Save logs and plot curves
        with open(output_dir / "log.json", "w") as f:
            json.dump(history, f, indent=4)
        
        plot_cifar_metrics(
            history["train_loss"], 
            history["nll"], 
            save_path=str(output_dir / "training_metrics.png")
        )

    total_time = time.time() - start_time
    print(f"Training finished in {total_time/3600:.2f} hours")
    
    # Save final model separately
    torch.save(model.state_dict(), output_dir / "model_final.pt")
    print(f"Final model saved to {output_dir}/model_final.pt")


if __name__ == "__main__":
    args = parse_args()
    train(args)