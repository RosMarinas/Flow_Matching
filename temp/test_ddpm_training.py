"""
Quick test script to verify DDPM training works correctly.
Trains for only 10 epochs to validate the pipeline.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test 2D DDPM training (10 epochs)
print("=" * 60)
print("Testing 2D DDPM Training (10 epochs)")
print("=" * 60)

import subprocess
result = subprocess.run([
    "uv", "run", "src/train_toy_ddpm.py",
    "--hidden_dim", "128",
    "--num_layers", "3",
    "--epochs", "10",
    "--dataset", "checkerboard",
    "--num_timesteps", "1000",
    "--beta_schedule", "linear",
    "--output_dir", "temp/test_ddpm_2d",
    "--device", "cuda"
])

if result.returncode == 0:
    print("\n✅ 2D DDPM training test PASSED!")
    print("Ready for full training on server.")
else:
    print("\n❌ 2D DDPM training test FAILED!")
    print("Please check the errors above.")

sys.exit(result.returncode)
