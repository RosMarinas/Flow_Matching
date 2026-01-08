"""
Sample DDPM models from different epochs

This script loads DDPM checkpoints from results/cifar10/DDPM/checkpoints/
and generates samples with fixed NFE=100 for each epoch.
"""

import sys
from pathlib import Path
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import UNet
from src.ddpm_solver import ddim_sample
from src.visualize import save_samples


def main():
    print("=" * 60)
    print("Sampling DDPM Models from Different Epochs")
    print("=" * 60)

    # Configuration - MUST match training config!
    model_channels = 32
    num_res_blocks = 2
    num_timesteps = 1000  # From training
    beta_schedule = "cosine"  # CRITICAL: Must match training!
    nfe = 100  # Fixed NFE for sampling
    num_samples = 64  # Number of samples to generate per epoch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDevice: {device}")
    print(f"Model config: channels={model_channels}, blocks={num_res_blocks}")
    print(f"DDPM config: timesteps={num_timesteps}, schedule={beta_schedule}")
    print(f"Sampling NFE: {nfe}")

    # Find all checkpoint files
    checkpoint_dir = Path("results/cifar10/DDPM/checkpoints")

    if not checkpoint_dir.exists():
        print(f"[ERROR] Checkpoint directory not found: {checkpoint_dir}")
        return

    # Get all model_epoch*.pt files and sort by epoch number
    checkpoint_files = sorted(checkpoint_dir.glob("model_epoch*.pt"),
                              key=lambda x: int(x.stem.split("epoch")[1]))

    if not checkpoint_files:
        print(f"[ERROR] No checkpoints found in {checkpoint_dir}")
        return

    print(f"\nFound {len(checkpoint_files)} checkpoint files:")
    for ckpt in checkpoint_files:
        print(f"  - {ckpt.name}")

    # Create output directory
    output_dir = Path("results/cifar10/DDPM/samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    model = UNet(
        in_channels=3,
        out_channels=3,
        model_channels=model_channels,
        num_res_blocks=num_res_blocks,
        channel_mult=(1, 2, 2, 2),
        attention_resolutions=(2,),
        dropout=0.1,
        num_heads=4,
        use_discrete_time=True,  # Critical for DDPM!
        max_timesteps=num_timesteps
    ).to(device)

    # Process each checkpoint
    for checkpoint_path in checkpoint_files:
        epoch_num = checkpoint_path.stem.split("epoch")[1]
        print(f"\n{'=' * 60}")
        print(f"Processing Epoch {epoch_num}")
        print(f"{'=' * 60}")

        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path.name}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load weights (prefer EMA if available)
        if 'model_state_dict' in checkpoint:
            if 'ema_shadow' in checkpoint:
                print("  Using EMA weights for better quality...")
                model.load_state_dict(checkpoint['ema_shadow'])
                print(f"  Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.6f}")
            else:
                print("  Using model weights...")
                model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("  Using model weights...")
            model.load_state_dict(checkpoint)

        model.eval()

        # Generate samples
        print(f"Generating {num_samples} samples with NFE={nfe}...")
        with torch.no_grad():
            samples = ddim_sample(
                model,
                num_samples=num_samples,
                input_shape=(3, 32, 32),
                num_steps=nfe,
                eta=0.0,
                device=device,
                num_timesteps=num_timesteps,
                schedule=beta_schedule
            )

        # Check sample statistics
        print(f"Sample Statistics:")
        print(f"  Shape: {samples.shape}")
        print(f"  Min: {samples.min().item():.4f}")
        print(f"  Max: {samples.max().item():.4f}")
        print(f"  Mean: {samples.mean().item():.4f}")
        print(f"  Std: {samples.std().item():.4f}")

        # Check for NaN/Inf
        if torch.isnan(samples).any():
            print("[ERROR] Samples contain NaN! Skipping this checkpoint.")
            continue
        if torch.isinf(samples).any():
            print("[ERROR] Samples contain Inf! Skipping this checkpoint.")
            continue

        # Save samples
        save_path = output_dir / f"samples_epoch{epoch_num}_nfe{nfe}.png"
        save_samples(samples, str(save_path), nrow=8, normalize=True, value_range=(-1, 1))
        print(f"[OK] Samples saved: {save_path}")

    print(f"\n{'=' * 60}")
    print("[SUCCESS] All checkpoints processed!")
    print(f"{'=' * 60}")
    print(f"\nGenerated samples saved to: {output_dir}/")
    print(f"Total: {len(checkpoint_files)} epoch checkpoints × {num_samples} samples each")


if __name__ == "__main__":
    main()
