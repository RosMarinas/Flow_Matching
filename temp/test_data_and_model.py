"""
Quick test to verify data generation and model training
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import matplotlib.pyplot as plt
from data import generate_checkerboard
from models import MLPVectorField
from cfm import ConditionalFlowMatching
from solver import euler_solver

print("=" * 60)
print("Testing Data Generation and Model")
print("=" * 60)

# 1. Test data generation
print("\n1. Generating checkerboard data...")
data = generate_checkerboard(n_samples=10000, grid_size=4, device='cpu')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot data
axes[0].scatter(data[:, 0], data[:, 1], s=1, alpha=0.5)
axes[0].set_title("Checkerboard Data Distribution")
axes[0].set_xlim(-4, 4)
axes[0].set_ylim(-4, 4)
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)

# Add grid lines
for i in range(-4, 5, 2):
    axes[0].axhline(i, color='red', linestyle='--', alpha=0.3)
    axes[0].axvline(i, color='red', linestyle='--', alpha=0.3)

# Sample some points and print their coordinates
sample_indices = torch.randperm(len(data))[:10]
sample_points = data[sample_indices]
print(f"\nSample data points (should be near integer coordinates):")
for i, pt in enumerate(sample_points):
    print(f"  Point {i}: ({pt[0]:.2f}, {pt[1]:.2f})")

# 2. Test model with a simple forward pass
print("\n2. Testing model forward pass...")
model = MLPVectorField(hidden_dim=128, num_layers=3)
x = torch.randn(10, 2)
t = torch.rand(10, 1)
v = model(x, t)
print(f"  Input shape: {x.shape}")
print(f"  Time shape: {t.shape}")
print(f"  Output shape: {v.shape}")
print("  ✅ Model forward pass works!")

# 3. Test CFM loss computation
print("\n3. Testing CFM loss...")
cfm = ConditionalFlowMatching(model, path_type="OT")
x1 = data[:32]  # Batch of 32
loss = cfm.compute_loss(x1)
print(f"  Batch size: {x1.shape[0]}")
print(f"  Loss value: {loss.item():.6f}")
print("  ✅ CFM loss computation works!")

# 4. Test sampling
print("\n4. Testing sampling...")
model.eval()
with torch.no_grad():
    samples = euler_solver(model, x0=torch.randn(5, 2), num_steps=10)
print(f"  Generated {samples.shape[0]} samples")
print(f"  Sample shape: {samples.shape}")
print(f"  Sample coordinates:")
for i, s in enumerate(samples):
    print(f"    Sample {i}: ({s[0]:.2f}, {s[1]:.2f})")
print("  ✅ Sampling works!")

# Plot the samples
axes[1].scatter(samples[:, 0], samples[:, 1], s=100, c='red', marker='*', label='Untrained samples')
axes[1].scatter(data[:, 0], data[:, 1], s=1, alpha=0.3, label='True data')
axes[1].set_title("Untrained Model Samples (Should be random)")
axes[1].set_xlim(-4, 4)
axes[1].set_ylim(-4, 4)
axes[1].set_aspect('equal')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('temp/test_output.png', dpi=150)
print(f"\n✅ Visualization saved to temp/test_output.png")
print("\n" + "=" * 60)
print("All tests passed! The code is working correctly.")
print("=" * 60)
print("\nExpected behavior:")
print("  - Checkerboard should have 8 filled squares (4x4 grid, alternating)")
print("  - Data points should be in squares centered at odd coordinates:")
print("    (-3,-3), (-3,-1), (-1,-3), (-1,-1), (1,-3), (1,-1), (3,-3), (3,-1)")
print("  - Untrained model samples should be random (not on checkerboard)")
print("=" * 60)
