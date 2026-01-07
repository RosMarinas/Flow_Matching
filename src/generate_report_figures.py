"""
Report Figure Generation Script

Generates all figures needed for the Project 3 report by aggregating
results from experiments and calling visualization functions.

Usage:
    uv run src/generate_report_figures.py --output_dir report/figures
"""

import sys
from pathlib import Path
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import UNet, MLPVectorField
from src.solver import euler_solver
from src.visualize import (
    plot_vector_field,
    plot_trajectories,
    compare_paths,
    plot_flow_evolution,
    plot_training_curves,
    plot_nfe_curve,
    save_samples
)


def generate_2d_toy_figures(
    checkpoint_ot='results/toy/checkpoints/model_ot.pt',
    checkpoint_vp='results/toy/checkpoints/model_vp.pt',
    output_dir='report/figures/toy'
):
    """
    Generate all 2D toy comparison figures.

    This creates the key visualizations showing OT's straight-line
    trajectories vs VP's curved trajectories.
    """
    print("\n" + "="*60)
    print("Generating 2D Toy Comparison Figures")
    print("="*60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load models
    print("Loading models...")
    print("Using hidden_dim=512, num_layers=5 (from checkpoint)")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_ot = MLPVectorField(hidden_dim=512, num_layers=5).to(device)
    model_vp = MLPVectorField(hidden_dim=512, num_layers=5).to(device)

    if Path(checkpoint_ot).exists():
        model_ot.load_state_dict(torch.load(checkpoint_ot, map_location=device))
        print(f"✅ Loaded OT model from {checkpoint_ot}")
    else:
        print(f"⚠️  OT checkpoint not found: {checkpoint_ot}")
        return False

    if Path(checkpoint_vp).exists():
        model_vp.load_state_dict(torch.load(checkpoint_vp, map_location=device))
        print(f"✅ Loaded VP model from {checkpoint_vp}")
    else:
        print(f"⚠️  VP checkpoint not found: {checkpoint_vp}")
        return False

    model_ot.eval()
    model_vp.eval()

    # Generate comparison figure (KEY FIGURE!)
    print("\nGenerating path comparison...")
    compare_paths(
        model_ot,
        model_vp,
        save_path=str(output_path / "ot_vs_vp_trajectories.png"),
    )
    print("✅ Saved: ot_vs_vp_trajectories.png")

    # Generate vector fields
    print("\nGenerating vector fields...")
    plot_vector_field(
        model_ot,
        t_values=[0.0, 0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        save_path=str(output_path / "vector_field_OT.png"),
        title_prefix="OT Path - ",
    )
    print("✅ Saved: vector_field_OT.png")

    plot_vector_field(
        model_vp,
        t_values=[0.0, 0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        save_path=str(output_path / "vector_field_VP.png"),
        title_prefix="VP Path - ",
    )
    print("✅ Saved: vector_field_VP.png")

    # Generate flow evolution
    print("\nGenerating flow evolution...")
    plot_flow_evolution(
        model_ot,
        euler_solver,
        t_values=[0.0, 0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        n_samples=10000,
        save_path=str(output_path / "flow_evolution_OT.png"),
        title="OT Path - Flow Evolution"
    )
    print("✅ Saved: flow_evolution_OT.png")

    plot_flow_evolution(
        model_vp,
        euler_solver,
        t_values=[0.0, 0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        n_samples=10000,
        save_path=str(output_path / "flow_evolution_VP.png"),
        title="VP Path - Flow Evolution"
    )
    print("✅ Saved: flow_evolution_VP.png")

    # Load training logs if available
    print("\nLooking for training logs...")
    log_ot = Path('results/toy/checkpoints/training_log_ot.json')
    log_vp = Path('results/toy/checkpoints/training_log_vp.json')

    if log_ot.exists() and log_vp.exists():
        with open(log_ot, 'r') as f:
            data_ot = json.load(f)
        with open(log_vp, 'r') as f:
            data_vp = json.load(f)

        losses_ot = data_ot.get('train_losses', [])
        losses_vp = data_vp.get('train_losses', [])

        if losses_ot and losses_vp:
            print("Generating training curves...")
            plot_training_curves(
                losses_ot,
                losses_vp,
                save_path=str(output_path / "training_curves_comparison.png"),
            )
            print("✅ Saved: training_curves_comparison.png")

    print(f"\n✅ All 2D toy figures saved to {output_path}/")
    return True


def generate_cifar10_comparison_table(
    nfe_results_file='results/cifar10/nfe_sweep/nfe_comparison_results.json',
    output_dir='report/tables'
):
    """
    Generate Table 1 comparison table from NFE sweep results.
    """
    print("\n" + "="*60)
    print("Generating CIFAR-10 Comparison Table")
    print("="*60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = Path(nfe_results_file)
    if not results_file.exists():
        print(f"⚠️  Results file not found: {results_file}")
        print("   Run NFE sweep first: uv run src/nfe_sweep.py --checkpoint_ot ... --checkpoint_vp ...")
        return False

    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Extract FID scores (use NFE=100 results)
    table_data = []

    # OT results
    if 'ot' in results:
        ot_data = results['ot']
        nfe_idx = ot_data['nfe_values'].index(100) if 100 in ot_data['nfe_values'] else 0
        fid_ot = ot_data['fid_scores'][nfe_idx] if ot_data['fid_scores'] else None
        table_data.append({
            'Model': 'FM-OT (Ours)',
            'NLL': 'N/A',  # Would need separate NLL computation
            'FID': f'{fid_ot:.2f}' if fid_ot else 'N/A',
            'NFE': 100
        })

    # VP results
    if 'vp' in results:
        vp_data = results['vp']
        nfe_idx = vp_data['nfe_values'].index(100) if 100 in vp_data['nfe_values'] else 0
        fid_vp = vp_data['fid_scores'][nfe_idx] if vp_data['fid_scores'] else None
        table_data.append({
            'Model': 'FM-VP (Ours)',
            'NLL': 'N/A',
            'FID': f'{fid_vp:.2f}' if fid_vp else 'N/A',
            'NFE': 100
        })

    # Add paper baseline data for comparison (from Table 1 of paper)
    table_data.extend([
        {'Model': 'DDPM (Paper)', 'NLL': '3.12', 'FID': '7.48', 'NFE': 274},
        {'Model': 'Score Matching (Paper)', 'NLL': '3.16', 'FID': '19.94', 'NFE': 242},
        {'Model': 'FM-OT (Paper)', 'NLL': '2.99', 'FID': '6.35', 'NFE': 142},
    ])

    # Generate markdown table
    table_md = "| Model | NLL | FID | NFE |\n"
    table_md += "|-------|-----|-----|-----|\n"
    for row in table_data:
        table_md += f"| {row['Model']} | {row['NLL']} | {row['FID']} | {row['NFE']} |\n"

    # Save markdown table with UTF-8 encoding
    table_file = output_path / 'table1_comparison.md'
    with open(table_file, 'w', encoding='utf-8') as f:
        f.write("# CIFAR-10 Results Comparison (Table 1)\n\n")
        f.write(table_md)

    print(f"\n✅ Comparison table saved to {table_file}")
    print("\nGenerated Table:")
    print(table_md)

    return True


def generate_nfe_comparison_figure(
    nfe_results_file='results/cifar10/nfe_sweep/nfe_comparison_results.json',
    output_dir='report/figures/cifar10'
):
    """
    Generate FID vs NFE comparison curve from NFE sweep results.
    """
    print("\n" + "="*60)
    print("Generating NFE Comparison Figure")
    print("="*60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = Path(nfe_results_file)
    if not results_file.exists():
        print(f"⚠️  Results file not found: {results_file}")
        return False

    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Extract data - handle both formats
    # Format 1: {'ot': {...}, 'vp': {...}}
    # Format 2: {'nfe_values': [...], 'ot_fid': [...], 'vp_fid': [...]}
    if 'ot' in results and 'vp' in results:
        nfe_ot = results['ot'].get('nfe_values', [])
        fid_ot = results['ot'].get('fid_scores', [])
        nfe_vp = results['vp'].get('nfe_values', [])
        fid_vp = results['vp'].get('fid_scores', [])
    elif 'nfe_values' in results and 'ot_fid' in results:
        nfe_ot = results['nfe_values']
        fid_ot = results['ot_fid']
        nfe_vp = results['nfe_values']
        fid_vp = results['vp_fid']
    else:
        print("⚠️  Unexpected JSON format")
        print(f"Keys: {list(results.keys())}")
        return False

    if not nfe_ot or not fid_ot or not nfe_vp or not fid_vp:
        print("⚠️  No NFE sweep data found")
        print(f"nfe_ot: {len(nfe_ot) if nfe_ot else 0}, fid_ot: {len(fid_ot) if fid_ot else 0}")
        return False

    print(f"Data loaded: OT ({len(nfe_ot)} points), VP ({len(nfe_vp)} points)")

    # Generate comparison plot
    plot_nfe_curve(
        nfe_list=nfe_ot,
        fid_scores=fid_ot,
        label="FM-OT",
        compare_nfe=nfe_vp,
        compare_fid=fid_vp,
        compare_label="FM-VP",
        save_path=str(output_path / 'nfe_comparison.png')
    )

    print(f"✅ NFE comparison figure saved to {output_path / 'nfe_comparison.png'}")
    return True


def generate_sample_comparison(
    checkpoint_ot='results/cifar10/OT/model_final.pt',
    checkpoint_vp='results/cifar10/VP/model_final.pt',
    output_dir='report/figures/cifar10',
    num_samples=64,
    device='cuda'
):
    """
    Generate side-by-side sample comparison grid.
    """
    print("\n" + "="*60)
    print("Generating Sample Comparison Grid")
    print("="*60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load models
    print("Loading models...")
    try:
        checkpoint_ot_path = Path(checkpoint_ot)
        checkpoint_vp_path = Path(checkpoint_vp)

        if not checkpoint_ot_path.exists() or not checkpoint_vp_path.exists():
            print(f"⚠️  Checkpoints not found")
            return False

        # Load OT model
        print("Loading models with model_channels=32 (from checkpoint)")

        model_ot = UNet(
            in_channels=3, out_channels=3,
            model_channels=32, num_res_blocks=2,
            channel_mult=(1, 2, 2, 2),
            attention_resolutions=(2,),
            dropout=0.1, num_heads=4
        ).to(device)

        checkpoint_ot_data = torch.load(checkpoint_ot_path, map_location=device)
        state_dict = checkpoint_ot_data.get('model_state_dict', checkpoint_ot_data)
        model_ot.load_state_dict(state_dict)
        model_ot.eval()

        # Load VP model
        model_vp = UNet(
            in_channels=3, out_channels=3,
            model_channels=32, num_res_blocks=2,
            channel_mult=(1, 2, 2, 2),
            attention_resolutions=(2,),
            dropout=0.1, num_heads=4
        ).to(device)

        checkpoint_vp_data = torch.load(checkpoint_vp_path, map_location=device)
        state_dict = checkpoint_vp_data.get('model_state_dict', checkpoint_vp_data)
        model_vp.load_state_dict(state_dict)
        model_vp.eval()

        print("✅ Models loaded")

        # Generate samples
        print(f"\nGenerating {num_samples} samples per model...")

        with torch.no_grad():
            # OT samples
            x0 = torch.randn(num_samples, 3, 32, 32, device=device)
            samples_ot = euler_solver(model_ot, x0, num_steps=100)

            # VP samples
            x0 = torch.randn(num_samples, 3, 32, 32, device=device)
            samples_vp = euler_solver(model_vp, x0, num_steps=100)

        # Save comparison grid
        from torchvision.utils import make_grid
        import torchvision.io as io

        # Create side-by-side comparison
        # Top row: OT samples, Bottom row: VP samples
        grid_ot = make_grid(samples_ot, nrow=8, normalize=True, value_range=(-1, 1))
        grid_vp = make_grid(samples_vp, nrow=8, normalize=True, value_range=(-1, 1))

        # Stack vertically
        comparison_grid = torch.cat([grid_ot, grid_vp], dim=1)

        save_path = output_path / 'sample_comparison_ot_vs_vp.png'
        save_samples(comparison_grid.unsqueeze(0), str(save_path), nrow=1)

        print(f"✅ Sample comparison saved to {save_path}")
        return True

    except Exception as e:
        print(f"⚠️  Error generating samples: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate report figures")
    parser.add_argument('--output_dir', type=str, default='report/figures',
                        help='Output directory for figures')
    parser.add_argument('--generate_2d', action='store_true',
                        help='Generate 2D toy figures')
    parser.add_argument('--generate_cifar10', action='store_true',
                        help='Generate CIFAR-10 figures')
    parser.add_argument('--checkpoint_ot', type=str,
                        default='results/cifar10/OT/model_final.pt')
    parser.add_argument('--checkpoint_vp', type=str,
                        default='results/cifar10/VP/model_final.pt')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Generate all figures if no specific option given
    if not args.generate_2d and not args.generate_cifar10:
        args.generate_2d = True
        args.generate_cifar10 = True

    success = True

    if args.generate_2d:
        success &= generate_2d_toy_figures(
            output_dir=f"{args.output_dir}/toy"
        )

    if args.generate_cifar10:
        success &= generate_cifar10_comparison_table(
            output_dir=f"{args.output_dir}/../tables"
        )
        success &= generate_nfe_comparison_figure(
            output_dir=f"{args.output_dir}/cifar10"
        )
        success &= generate_sample_comparison(
            checkpoint_ot=args.checkpoint_ot,
            checkpoint_vp=args.checkpoint_vp,
            output_dir=f"{args.output_dir}/cifar10",
            device=args.device
        )

    print("\n" + "="*60)
    if success:
        print("✅ All figures generated successfully!")
    else:
        print("⚠️  Some figures failed to generate")
    print("="*60 + "\n")

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
