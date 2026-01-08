
"""
Quick Diagnostic Test for DDPM Configuration

Tests different timestep/sampling configurations to identify the issue.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.ddpm import DDPM
from src.ddpm_solver import ddim_sample
from src.models import UNet


def test_sampling_config(num_timesteps, num_ddim_steps, model_path, device="cuda"):
    """Test a specific configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: num_timesteps={num_timesteps}, DDIM steps={num_ddim_steps}")
    print(f"Stride: {num_timesteps/num_ddim_steps:.1f}")
    print(f"{'='*60}")

    # Load model (use the already trained weights)
    model = UNet(
        in_channels=3,
        out_channels=3,
        model_channels=32,
        num_res_blocks=2,
        channel_mult=(1, 2, 2, 2),
        attention_resolutions=(2,),
        dropout=0.1,
        num_heads=4,
        use_discrete_time=True,
        max_timesteps=num_timesteps
    ).to(device)

    # Load checkpoint (trained with 10000 timesteps)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"[OK] Loaded checkpoint: {model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        return None

    model.eval()

    # Create DDPM to get beta schedule
    ddpm = DDPM(model, num_timesteps=num_timesteps, beta_schedule="cosine")

    # Test sampling with different DDIM steps
    with torch.no_grad():
        try:
            samples = ddim_sample(
                model,
                num_samples=16,
                input_shape=(3, 32, 32),
                num_steps=num_ddim_steps,
                eta=0.0,
                device=device,
                num_timesteps=num_timesteps
            )

            print(f"[OK] Sampling successful!")
            print(f"  Sample stats: min={samples.min().item():.4f}, max={samples.max().item():.4f}, mean={samples.mean().item():.4f}")
            print(f"  Std: {samples.std().item():.4f}")

            return samples
        except Exception as e:
            print(f"[ERROR] Sampling failed: {e}")
            return None


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="results/cifar10/DDPM/checkpoints/model_epoch10.pt",
                        help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("DDPM Configuration Diagnostic Test")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")

    # Test configurations
    configs = [
        # (num_timesteps, num_ddim_steps, description)
        (10000, 100, "Original: 10k timesteps, 100 DDIM steps (stride=100)"),
        (10000, 1000, "10k timesteps, 1000 DDIM steps (stride=10)"),
        (1000, 100, "Standard: 1k timesteps, 100 DDIM steps (stride=10)"),
        (1000, 250, "1k timesteps, 250 DDIM steps (stride=4)"),
    ]

    results = []
    for num_timesteps, num_ddim_steps, desc in configs:
        samples = test_sampling_config(num_timesteps, num_ddim_steps, args.checkpoint, args.device)
        results.append((desc, samples))

    # Visualize results
    print("\n" + "="*60)
    print("Generating comparison visualization...")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for idx, (desc, samples) in enumerate(results):
        ax = axes[idx]

        if samples is not None:
            # Convert to grid for visualization
            grid = make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))

            # Transpose for matplotlib: (C, H, W) -> (H, W, C)
            grid = grid.cpu().numpy().transpose(1, 2, 0)

            ax.imshow(grid)
            ax.set_title(desc, fontsize=10)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, "SAMPLING FAILED", ha='center', va='center',
                   fontsize=14, color='red', transform=ax.transAxes)
            ax.set_title(desc, fontsize=10)
            ax.axis('off')

    plt.suptitle("DDPM Sampling Configuration Test (Epoch 10 Checkpoint)",
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = "temp/test_ddpm_config/comparison.png"
    Path("temp/test_ddpm_config").mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Comparison saved: {output_path}")

    plt.show()
