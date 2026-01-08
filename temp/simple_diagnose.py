
import torch
import sys
import os

# Ensure we can find src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Project root added: {project_root}")

try:
    from src.models import UNet
    print("Imported UNet")
except ImportError as e:
    print(f"Failed to import UNet: {e}")
    sys.exit(1)

def main():
    ckpt_path = os.path.join(project_root, "results", "cifar10", "DDPM", "checkpoints", "model_final_1000epoch.pt")
    if not os.path.exists(ckpt_path):
        # Try checking root of results
        ckpt_path_alt = os.path.join(project_root, "results", "cifar10", "DDPM", "model_final_1000epoch.pt")
        if os.path.exists(ckpt_path_alt):
            ckpt_path = ckpt_path_alt
        else:
            print(f"Checkpoint not found: {ckpt_path}")
            # List available
            d = os.path.join(project_root, "results", "cifar10", "DDPM", "checkpoints")
            if os.path.exists(d):
                print(f"Contents of {d}: {os.listdir(d)}")
            return

    print(f"Loading {ckpt_path}...")
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        print("Checkpoint loaded.")
        if isinstance(checkpoint, dict):
            print(f"Keys: {list(checkpoint.keys())}")
            if 'loss' in checkpoint:
                print(f"Loss: {checkpoint['loss']}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    main()
