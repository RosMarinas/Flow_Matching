"""
Generate class-labeled samples from trained Flow Matching model.

Usage:
    uv run src/sample_cifar_with_labels.py \
        --checkpoint results/cifar10/with_labels/OT/model_final.pt \
        --num_samples 16 \
        --num_steps 50
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import argparse
from src.models import UNetWithClassLabels
from src.visualize import save_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Generate class-labeled samples from trained Flow Matching model")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples per class")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of ODE integration steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="results/cifar10/with_labels/samples", help="Output directory")
    parser.add_argument("--model_channels", type=int, default=32, help="Model channels (must match training)")

    return parser.parse_args()


@torch.no_grad()
def sample_with_labels(model, class_labels, num_steps, device):
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


def main(args):
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Class-Labeled CIFAR-10 Sample Generation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Samples per class: {args.num_samples}")
    print(f"ODE steps: {args.num_steps}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Create model with the same architecture as training
    model = UNetWithClassLabels(
        num_classes=10,
        class_emb_dim=256,
        in_channels=3,
        out_channels=3,
        model_channels=args.model_channels,
        num_res_blocks=2,
        channel_mult=(1, 2, 2, 2),
        attention_resolutions=(2,),
        dropout=0.1,
        num_heads=4
    ).to(args.device)

    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict")

    model.eval()

    # Generate samples for each class
    print(f"\nGenerating {args.num_samples} samples for each of 10 classes...")
    all_samples = []

    for class_id in range(10):
        print(f"Generating samples for class {class_id}...")
        class_labels = torch.full((args.num_samples,), class_id, dtype=torch.long, device=args.device)

        samples = sample_with_labels(model, class_labels, args.num_steps, args.device)
        all_samples.append(samples)

        # Save per-class samples
        save_path = output_dir / f"class_{class_id}_samples.png"
        save_samples(samples, str(save_path), nrow=int(args.num_samples**0.5), normalize=True, value_range=(-1, 1))
        print(f"  Saved to {save_path}")

    # Save combined grid (10 rows x num_samples columns)
    all_samples = torch.cat(all_samples, dim=0)  # (10*num_samples, 3, 32, 32)
    save_path = output_dir / "all_classes_grid.png"
    save_samples(all_samples, str(save_path), nrow=args.num_samples, normalize=True, value_range=(-1, 1))
    print(f"\nSaved combined grid to {save_path}")

    print("\n" + "=" * 60)
    print("Sample generation complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    main(args)
