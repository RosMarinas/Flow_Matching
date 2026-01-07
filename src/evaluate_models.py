"""
Simple Model Evaluation Script

Runs evaluation for trained models using existing infrastructure.
Maximizes code reuse by calling nfe_sweep and train_cifar functions.

Usage:
    # Evaluate single model
    uv run src/evaluate_models.py --checkpoint results/cifar10/OT/model_final.pt

    # Evaluate both OT and VP
    uv run src/evaluate_models.py --checkpoint_ot results/cifar10/OT/model_final.pt \
                                    --checkpoint_vp results/cifar10/VP/model_final.pt
"""

import sys
from pathlib import Path
import json
import subprocess

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def evaluate_single_model(checkpoint_path, num_samples=10000, device='cuda'):
    """
    Evaluate a single model using nfe_sweep infrastructure.

    This runs NFE sweep at a single NFE value (100 steps) to get FID.
    """
    checkpoint_path = Path(checkpoint_path)

    print(f"\n{'='*60}")
    print(f"Evaluating: {checkpoint_path.name}")
    print(f"{'='*60}\n")

    # Determine path type from checkpoint path
    path_type = 'OT' if 'OT' in str(checkpoint_path) or 'ot' in str(checkpoint_path) else 'VP'

    # Run nfe_sweep for this model at NFE=100
    cmd = [
        'uv', 'run', 'src/nfe_sweep.py',
        f'--checkpoint_{path_type.lower()}', str(checkpoint_path),
        '--num_samples', str(num_samples),
        '--device', device
    ]

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=True)

    return result.returncode == 0


def evaluate_both_models(checkpoint_ot, checkpoint_vp, num_samples=10000, device='cuda'):
    """
    Evaluate both OT and VP models and generate comparison.
    """
    print("\n" + "="*60)
    print("Evaluating both OT and VP models")
    print("="*60 + "\n")

    # Run nfe_sweep for both models
    cmd = [
        'uv', 'run', 'src/nfe_sweep.py',
        '--checkpoint_ot', str(checkpoint_ot),
        '--checkpoint_vp', str(checkpoint_vp),
        '--num_samples', str(num_samples),
        '--device', device
    ]

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=True)

    # Load and display results
    results_file = Path('results/cifar10/nfe_sweep/nfe_comparison_results.json')
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)

        print("\n" + "="*60)
        print("Evaluation Summary")
        print("="*60)

        # Display OT results
        if 'ot' in results:
            ot_results = results['ot']
            print(f"\nOT Path (NFE={ot_results['nfe_values'][0] if ot_results['nfe_values'] else 'N/A'}):")
            print(f"  FID: {ot_results['fid_scores'][0]:.4f}" if ot_results['fid_scores'] else "  FID: N/A")

        # Display VP results
        if 'vp' in results:
            vp_results = results['vp']
            print(f"\nVP Path (NFE={vp_results['nfe_values'][0] if vp_results['nfe_values'] else 'N/A'}):")
            print(f"  FID: {vp_results['fid_scores'][0]:.4f}" if vp_results['fid_scores'] else "  FID: N/A")

        print("\n" + "="*60)

    return result.returncode == 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained Flow Matching models")
    parser.add_argument('--checkpoint', type=str, help='Path to single model checkpoint')
    parser.add_argument('--checkpoint_ot', type=str, help='Path to OT model checkpoint')
    parser.add_argument('--checkpoint_vp', type=str, help='Path to VP model checkpoint')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples for FID')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')

    args = parser.parse_args()

    if args.checkpoint:
        # Evaluate single model
        success = evaluate_single_model(args.checkpoint, args.num_samples, args.device)
    elif args.checkpoint_ot and args.checkpoint_vp:
        # Evaluate both models
        success = evaluate_both_models(args.checkpoint_ot, args.checkpoint_vp, args.num_samples, args.device)
    else:
        print("Error: Must provide either --checkpoint or both --checkpoint_ot and --checkpoint_vp")
        return 1

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
