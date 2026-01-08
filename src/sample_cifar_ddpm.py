"""
CIFAR-10 DDPM Sampling Script

This script uses the trained DDPM model to generate samples with different NFE values
using DDIM sampling for variable computational cost.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.models import UNet
from src.ddpm_solver import ddim_sample
from src.visualize import save_samples


def main():
    print("=" * 60)
    print("CIFAR-10 DDPM Sampling with Cosine Schedule")
    print("=" * 60)

    # Configuration - MUST match training config!
    model_channels = 32
    num_res_blocks = 2
    num_timesteps = 1000  # From training
    beta_schedule = "cosine"  # CRITICAL: Must match training!
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDevice: {device}")
    print(f"Model config: channels={model_channels}, blocks={num_res_blocks}")
    print(f"DDPM config: timesteps={num_timesteps}, schedule={beta_schedule}")

    # Load model
    print("\nLoading trained model...")
    checkpoint_path = Path("results/cifar10/DDPM/checkpoints/model_epoch1000.pt")

    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("Please train the model first with:")
        print("  uv run src/train_cifar_ddpm.py --epochs 1000")
        return

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

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Check if this is a full training checkpoint or just model weights
    if 'model_state_dict' in checkpoint:
        # Full training checkpoint - use EMA weights if available (better quality)
        print(f"  Detected full training checkpoint")
        if 'ema_shadow' in checkpoint:
            print(f"  Loading EMA weights (better quality)...")
            model.load_state_dict(checkpoint['ema_shadow'])
            print(f"  Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.6f}")
        else:
            print(f"  Loading model weights...")
            model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Just model state dict
        print(f"  Detected model weights only")
        model.load_state_dict(checkpoint)

    model.eval()

    print(f"[OK] Model loaded from {checkpoint_path}")

    # Test sampling with different NFE values
    nfe_values = [300, 500, 1000]

    for nfe in nfe_values:
        print(f"\n{'=' * 60}")
        print(f"Sampling with NFE={nfe}...")
        print(f"{'=' * 60}")

        with torch.no_grad():
            samples = ddim_sample(
                model,
                num_samples=64,
                input_shape=(3, 32, 32),
                num_steps=nfe,
                eta=0.0,
                device=device,
                num_timesteps=num_timesteps,
                schedule=beta_schedule  # CRITICAL: Must match training!
            )

        # Check sample statistics
        print(f"\nSample Statistics:")
        print(f"  Shape: {samples.shape}")
        print(f"  Min: {samples.min().item():.4f}")
        print(f"  Max: {samples.max().item():.4f}")
        print(f"  Mean: {samples.mean().item():.4f}")
        print(f"  Std: {samples.std().item():.4f}")

        # Check for NaN/Inf
        if torch.isnan(samples).any():
            print("[ERROR] Samples contain NaN! Sampling failed.")
            return
        if torch.isinf(samples).any():
            print("[ERROR] Samples contain Inf! Sampling failed.")
            return

        # Save samples
        output_dir = Path("results/cifar10/DDPM/test_samples")
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = output_dir / f"samples_nfe{nfe}.png"
        save_samples(samples, str(save_path), nrow=8, normalize=True, value_range=(-1, 1))
        print(f"[OK] Samples saved to {save_path}")

    print(f"\n{'=' * 60}")
    print("[SUCCESS] All sampling tests passed!")
    print(f"{'=' * 60}")
    print("\nGenerated samples are in: results/cifar10/DDPM/test_samples/")
    print("\nYou can view them to verify the model is working correctly.")


if __name__ == "__main__":
    main()
