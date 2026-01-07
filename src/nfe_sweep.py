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
from src.data import get_cifar10_dataloader
from src.metrics import compute_fid_from_files
from src.visualize import plot_nfe_curve


def parse_args():
    parser = argparse.ArgumentParser(description="Run NFE Sweep for Flow Matching")
    parser.add_argument("--checkpoint_ot", type=str, help="Path to OT model checkpoint")
    parser.add_argument("--checkpoint_vp", type=str, help="Path to VP model checkpoint")
    parser.add_argument("--model_channels", type=int, default=32, help="Model channels (must match checkpoint)")
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of samples for FID (10k+ recommended for final)")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for generation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="results/cifar10/nfe_sweep", help="Output directory")
    parser.add_argument("--ref_dir", type=str, default="data/cifar10_test_images", help="Directory to store reference images")
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


def main():
    args = parse_args()
    
    if not args.checkpoint_ot and not args.checkpoint_vp:
        print("Error: Must provide at least one of --checkpoint_ot or --checkpoint_vp")
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
        "vp_fid": None
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

    # 5. Save and Plot Results
    with open(output_dir / "nfe_comparison_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    plot_nfe_curve(
        nfe_list=nfe_values,
        fid_scores=results["ot_fid"] if results["ot_fid"] else [],
        label="OT Path",
        save_path=str(output_dir / "nfe_comparison_curve.png"),
        compare_nfe=nfe_values if results["vp_fid"] else None,
        compare_fid=results["vp_fid"],
        compare_label="VP Path"
    )
    
    print("\n✅ NFE sweep completed!")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
