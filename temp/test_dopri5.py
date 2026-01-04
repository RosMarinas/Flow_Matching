
import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import MLPVectorField
from src.solver import sample

def test_dopri5():
    print("Testing dopri5 integration...")
    
    # 1. Check if torchdiffeq is installed
    try:
        import torchdiffeq
        print(f"✅ torchdiffeq found (version: {torchdiffeq.__version__})")
    except ImportError:
        print("❌ torchdiffeq NOT found. Please run 'uv sync'.")
        return

    # 2. Create dummy model
    model = MLPVectorField(hidden_dim=32, num_layers=2)
    device = "cpu" # Test on CPU for simplicity
    
    # 3. Run sampling with dopri5
    print("\nRunning sample(method='dopri5')...")
    try:
        samples = sample(
            model, 
            num_samples=10, 
            input_shape=(2,), 
            method="dopri5",
            device=device
        )
        print(f"✅ Sampling successful! Output shape: {samples.shape}")
    except Exception as e:
        print(f"❌ Sampling failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dopri5()
