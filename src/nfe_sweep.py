"""
NFE Efficiency Sweep Script

Analyzes the trade-off between sampling quality (FID) and computational cost (NFE).
Generates FID vs NFE curves.
"""

import sys
from pathlib import Path
import json
import shutil
import torch
from torchvision.utils import save_image
from tqdm import tqdm
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import UNet
from src.solver import euler_solver
from src.ddpm_solver import ddim_sample
from src.data import get_cifar10_dataloader
from src.metrics import compute_fid_from_files
from src.visualize import plot_nfe_curve


def parse_args():
    parser = argparse.ArgumentParser(description="Run NFE Sweep for Flow Matching + DDPM")
    parser.add_argument("--checkpoint_ot", type=str, help="Path to OT model checkpoint")
    parser.add_argument("--checkpoint_vp", type=str, help="Path to VP model checkpoint")
    parser.add_argument("--checkpoint_ddpm", type=str, help="Path to DDPM model checkpoint")
    parser.add_argument("--model_channels", type=int, default=32, help="Model channels (must match checkpoint)")
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of samples for FID (10k+ recommended for final)")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for generation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="results/cifar10/nfe_sweep", help="Output directory")
    parser.add_argument("--ref_dir", type=str, default="data/cifar10_test_images", help="Directory to store reference images")
    # DDPM-specific parameters
    parser.add_argument("--ddpm_num_timesteps", type=int, default=1000, help="DDPM num_timesteps (must match training)")
    parser.add_argument("--ddpm_beta_schedule", type=str, default="cosine", help="DDPM beta_schedule (must match training)")
    return parser.parse_args()


def prepare_reference_images(output_dir: str, num_images: int = 10000):
    """
    Extract CIFAR-10 test images to a folder for FID computation.
    """
    output_path = Path(output_dir)
    if output_path.exists() and len(list(output_path.glob("*.png"))) >= num_images:
        print(f"Reference images already exist in {output_dir}")
        return

    print(f"Extracting {num_images} reference images to {output_dir}...")
    output_path.mkdir(parents=True, exist_ok=True)
    
    dataloader = get_cifar10_dataloader(batch_size=100, train=False, download=True)
    
    count = 0
    for x, _ in tqdm(dataloader, desc="Extracting refs"):
        # x is [-1, 1], normalize to [0, 1] for saving
        x = (x + 1) / 2.0
        x = torch.clamp(x, 0, 1)
        
        for i in range(x.shape[0]):
            if count >= num_images:
                break
            save_image(x[i], output_path / f"ref_{count:05d}.png")
            count += 1
        if count >= num_images:
            break


def load_model(checkpoint_path, model_channels, device):
    print(f"Loading model from {checkpoint_path}...")
    print(f"Using model_channels={model_channels}")

    model = UNet(
        in_channels=3,
        out_channels=3,
        model_channels=model_channels,
        num_res_blocks=2,
        channel_mult=(1, 2, 2, 2),
        attention_resolutions=(2,),
        dropout=0.1,
        num_heads=4
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_ddpm_model(checkpoint_path, model_channels, num_timesteps, device):
    print(f"Loading DDPM model from {checkpoint_path}...")
    print(f"Using model_channels={model_channels}, num_timesteps={num_timesteps}")

    model = UNet(
        in_channels=3,
        out_channels=3,
        model_channels=model_channels,
        num_res_blocks=2,
        channel_mult=(1, 2, 2, 2),
        attention_resolutions=(2,),
        dropout=0.1,
        num_heads=4,
        use_discrete_time=True,  # Critical for DDPM!
        max_timesteps=num_timesteps
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Use EMA weights if available (better quality)
    if 'model_state_dict' in checkpoint:
        if 'ema_shadow' in checkpoint:
            print("  Using EMA weights for better quality...")
            model.load_state_dict(checkpoint['ema_shadow'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def run_sweep(model, nfe_values, num_samples, batch_size, device, ref_dir, output_dir, prefix=""):
    fid_results = []
    
    for nfe in nfe_values:
        print(f"\n--- {prefix} NFE = {nfe} ---")
        
        # Create temp dir for this NFE
        temp_dir = output_dir / f"samples_{prefix}_nfe_{nfe}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()
        
        # Generate samples
        n_generated = 0
        pbar = tqdm(total=num_samples, desc=f"Generating {prefix} (NFE={nfe})")
        
        while n_generated < num_samples:
            current_batch_size = min(batch_size, num_samples - n_generated)
            
            with torch.no_grad():
                x0 = torch.randn(current_batch_size, 3, 32, 32, device=device)
                samples = euler_solver(model, x0, num_steps=nfe)
                
                # Normalize [0, 1]
                samples = (samples + 1) / 2.0
                samples = torch.clamp(samples, 0, 1)
                
                for i in range(current_batch_size):
                    save_image(samples[i], temp_dir / f"gen_{n_generated+i:05d}.png")
            
            n_generated += current_batch_size
            pbar.update(current_batch_size)
        pbar.close()
        
        # Compute FID
        print("Computing FID...")
        try:
            fid = compute_fid_from_files(
                str(ref_dir),
                str(temp_dir),
                batch_size=50,
                device=device
            )
            print(f"{prefix} NFE: {nfe}, FID: {fid:.4f}")
            fid_results.append(fid)
        except Exception as e:
            print(f"FID computation failed: {e}")
            fid_results.append(float('nan'))
            
        # Cleanup samples to save space
        shutil.rmtree(temp_dir)
        
    return fid_results


def run_ddpm_sweep(model, nfe_values, num_samples, batch_size, device, ref_dir, output_dir, prefix="", num_timesteps=1000, beta_schedule="cosine"):
    """
    Run NFE sweep for DDPM using DDIM sampling.

    Args:
        model: DDPM model
        nfe_values: List of NFE values to test
        num_samples: Total number of samples to generate
        batch_size: Batch size for generation
        device: Device to use
        ref_dir: Reference images directory
        output_dir: Output directory
        prefix: Prefix for naming (e.g., "DDPM")
        num_timesteps: DDPM num_timesteps (must match training)
        beta_schedule: DDPM beta_schedule (must match training)
    """
    fid_results = []

    for nfe in nfe_values:
        print(f"\n--- {prefix} NFE = {nfe} ---")

        # Create temp dir for this NFE
        temp_dir = output_dir / f"samples_{prefix}_nfe_{nfe}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()

        # Generate samples
        n_generated = 0
        pbar = tqdm(total=num_samples, desc=f"Generating {prefix} (NFE={nfe})")

        while n_generated < num_samples:
            current_batch_size = min(batch_size, num_samples - n_generated)

            with torch.no_grad():
                # Use DDIM sampling for variable NFE
                samples = ddim_sample(
                    model,
                    num_samples=current_batch_size,
                    input_shape=(3, 32, 32),
                    num_steps=nfe,
                    eta=0.0,  # Pure DDIM (eta=0) for deterministic sampling
                    device=device,
                    num_timesteps=num_timesteps,
                    schedule=beta_schedule
                )

                # Normalize [0, 1]
                samples = (samples + 1) / 2.0
                samples = torch.clamp(samples, 0, 1)

                for i in range(current_batch_size):
                    save_image(samples[i], temp_dir / f"gen_{n_generated+i:05d}.png")

            n_generated += current_batch_size
            pbar.update(current_batch_size)
        pbar.close()

        # Compute FID
        print("Computing FID...")
        try:
            fid = compute_fid_from_files(
                str(ref_dir),
                str(temp_dir),
                batch_size=50,
                device=device
            )
            print(f"{prefix} NFE: {nfe}, FID: {fid:.4f}")
            fid_results.append(fid)
        except Exception as e:
            print(f"FID computation failed: {e}")
            fid_results.append(float('nan'))

        # Cleanup samples to save space
        shutil.rmtree(temp_dir)

    return fid_results


def main():
    args = parse_args()

    if not args.checkpoint_ot and not args.checkpoint_vp and not args.checkpoint_ddpm:
        print("Error: Must provide at least one of --checkpoint_ot, --checkpoint_vp, or --checkpoint_ddpm")
        return

    # 1. Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Prepare reference statistics
    ref_count = min(10000, args.num_samples)
    prepare_reference_images(args.ref_dir, num_images=ref_count)

    # 3. Define NFE values
    nfe_values = [10, 20, 30, 40, 50, 60, 80, 100]

    # 4. Run Sweeps
    results = {
        "nfe_values": nfe_values,
        "ot_fid": None,
        "vp_fid": None,
        "ddpm_fid": None
    }

    # Run OT Sweep
    if args.checkpoint_ot:
        model_ot = load_model(args.checkpoint_ot, args.model_channels, args.device)
        print("\n=== Running OT Path Sweep ===")
        results["ot_fid"] = run_sweep(
            model_ot, nfe_values, args.num_samples, args.batch_size,
            args.device, args.ref_dir, output_dir, prefix="OT"
        )
        # Clear memory
        del model_ot
        torch.cuda.empty_cache()

    # Run VP Sweep
    if args.checkpoint_vp:
        model_vp = load_model(args.checkpoint_vp, args.model_channels, args.device)
        print("\n=== Running VP Path Sweep ===")
        results["vp_fid"] = run_sweep(
            model_vp, nfe_values, args.num_samples, args.batch_size,
            args.device, args.ref_dir, output_dir, prefix="VP"
        )
        del model_vp
        torch.cuda.empty_cache()

    # Run DDPM Sweep
    if args.checkpoint_ddpm:
        model_ddpm = load_ddpm_model(
            args.checkpoint_ddpm, args.model_channels,
            args.ddpm_num_timesteps, args.device
        )
        print("\n=== Running DDPM Sweep ===")
        results["ddpm_fid"] = run_ddpm_sweep(
            model_ddpm, nfe_values, args.num_samples, args.batch_size,
            args.device, args.ref_dir, output_dir, prefix="DDPM",
            num_timesteps=args.ddpm_num_timesteps,
            beta_schedule=args.ddpm_beta_schedule
        )
        del model_ddpm
        torch.cuda.empty_cache()

    # 5. Save and Plot Results
    with open(output_dir / "nfe_comparison_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Plot comparison curve
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    if results["ot_fid"]:
        plt.plot(nfe_values, results["ot_fid"], marker='o', label='OT Path', linewidth=2)
    if results["vp_fid"]:
        plt.plot(nfe_values, results["vp_fid"], marker='s', label='VP Path', linewidth=2)
    if results["ddpm_fid"]:
        plt.plot(nfe_values, results["ddpm_fid"], marker='^', label='DDPM', linewidth=2)

    plt.xlabel('NFE (Neural Function Evaluations)', fontsize=14)
    plt.ylabel('FID (Fréchet Inception Distance)', fontsize=14)
    plt.title('FID vs NFE: OT vs VP vs DDPM', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "nfe_comparison_three_way.png", dpi=300)
    plt.close()

    print("\n✅ NFE sweep completed!")
    print(f"Results saved to {output_dir}")
    print(f"  - nfe_comparison_results.json")
    print(f"  - nfe_comparison_three_way.png")


if __name__ == "__main__":
    main()
