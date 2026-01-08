
import torch
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")


from src.models import UNet
from src.ddpm import DDPM
from src.ddpm_solver import ddim_sample

def diagnose():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Path to checkpoint
    # Based on listing, it seems to be here:
    ckpt_path = Path("results/cifar10/DDPM/checkpoints/model_final_1000epoch.pt")
    
    if not ckpt_path.exists():
        # Try finding any .pt file in checkpoints
        print(f"Checkpoint not found at {ckpt_path}")
        ckpts = list(Path("results/cifar10/DDPM/checkpoints").glob("*.pt"))
        if not ckpts:
            print("No checkpoints found!")
            return
        ckpt_path = ckpts[-1]
        print(f"Using alternative checkpoint: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    # Check content
    if isinstance(checkpoint, dict):
        print("Checkpoint keys:", checkpoint.keys())
        if 'loss' in checkpoint:
            print(f"Saved Loss: {checkpoint['loss']}")
        if 'epoch' in checkpoint:
            print(f"Saved Epoch: {checkpoint['epoch']}")
    else:
        print("Checkpoint is likely just state_dict (not a dict with metadata).")

    # Initialize model
    # Assuming standard config from training script
    model_channels = 32
    num_res_blocks = 2
    model = UNet(
        in_channels=3,
        out_channels=3,
        model_channels=model_channels,
        num_res_blocks=num_res_blocks,
        channel_mult=(1, 2, 2, 2),
        attention_resolutions=(2,),
        dropout=0.1,
        num_heads=4,
        use_discrete_time=True,
        max_timesteps=10000 
    ).to(device)

    # Load weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model_state_dict.")
    elif isinstance(checkpoint, dict) and 'ema_shadow' in checkpoint:
        # If it's a checkpoint with EMA, we might want to test EMA weights?
        # The script saves EMA weights into 'ema_shadow' but also copies them to model before saving final?
        # Let's check if this is the final save or intermediate.
        # Intermediate saves: ema_shadow has EMA, model has current weights.
        # Final save: model state dict IS EMA weights.
        pass
    else:
        model.load_state_dict(checkpoint)
        print("Loaded state_dict directly.")

    model.eval()

    # Generate samples
    print("Generating samples...")
    with torch.no_grad():
        # Use DDIM with 100 steps
        samples = ddim_sample(
            model,
            num_samples=16,
            input_shape=(3, 32, 32),
            num_steps=100,
            eta=0.0,
            device=device,
            num_timesteps=10000
        )

    # Analyze samples
    print("\nSample Statistics:")
    print(f"Min: {samples.min().item():.4f}")
    print(f"Max: {samples.max().item():.4f}")
    print(f"Mean: {samples.mean().item():.4f}")
    print(f"Std: {samples.std().item():.4f}")

    # Check if they are just noise
    # If std is close to 1/sqrt(12) (uniform) or 1 (normal) and structureless?
    # Hard to tell without seeing, but if min/max are clipped at -1/1, it's expected.
    # If they are all 0 or NaN, that's bad.
    
    if torch.isnan(samples).any():
        print("ERROR: Samples contain NaNs!")
    
    # Save a small figure
    # from src.visualize import save_samples
    # save_samples(samples, "temp/test_output.png")

if __name__ == "__main__":
    diagnose()
