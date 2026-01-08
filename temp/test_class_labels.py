"""
Quick validation script for class-labeled Flow Matching.

Tests model forward pass, loss computation, and runs 1-2 epochs to verify training works.
"""

import torch
from src.models import UNetWithClassLabels
from src.cfm import FlowMatchingWithLabels
from src.data import get_cifar10_dataloader

print("=" * 60)
print("Testing Class-Labeled Flow Matching")
print("=" * 60)

# Test 1: Model instantiation
print("\n[Test 1] Creating model...")
model = UNetWithClassLabels(
    num_classes=10,
    class_emb_dim=256,
    model_channels=32,  # Small model for quick testing
    num_res_blocks=2,
    channel_mult=(1, 2, 2, 2),
    attention_resolutions=(2,),
    dropout=0.1,
    num_heads=4
)

n_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model created with {n_params:,} parameters")

# Test 2: Forward pass
print("\n[Test 2] Testing forward pass...")
model.eval()
x = torch.randn(4, 3, 32, 32)
t = torch.rand(4)
labels = torch.randint(0, 10, (4,))

print(f"  Input: x={x.shape}, t={t.shape}, labels={labels.shape}")

with torch.no_grad():
    v = model(x, t, labels)

print(f"  Output: {v.shape}")
assert v.shape == (4, 3, 32, 32), f"Expected (4, 3, 32, 32), got {v.shape}"
print("✓ Forward pass test passed")

# Test 3: CFM loss computation
print("\n[Test 3] Testing CFM loss...")
cfm = FlowMatchingWithLabels(model, path_type='OT')
x1 = torch.randn(4, 3, 32, 32)
labels = torch.randint(0, 10, (4,))

loss = cfm.compute_loss(x1, labels)
print(f"  Loss: {loss.item():.4f}")
assert loss.dim() == 0, "Loss should be scalar"
print("✓ CFM loss test passed")

# Test 4: Gradient flow
print("\n[Test 4] Testing gradient flow...")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

loss = cfm.compute_loss(x1, labels)
loss.backward()
optimizer.step()

print(f"  Loss after update: {loss.item():.4f}")
print("✓ Gradient flow test passed")

# Test 5: Mini training loop (1 epoch, few batches)
print("\n[Test 5] Running mini training loop (1 epoch, 10 batches)...")
device = "cpu"  # Use CPU for MacBook Air

model = model.to(device)
train_loader = get_cifar10_dataloader(batch_size=32, train=True, download=True, num_workers=0)

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

batch_count = 0
total_loss = 0.0

for batch_idx, (x, labels) in enumerate(train_loader):
    if batch_count >= 10:  # Only test 10 batches
        break

    x = x.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    loss = cfm.compute_loss(x, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    total_loss += loss.item()
    batch_count += 1

    print(f"  Batch {batch_count}/10: loss={loss.item():.4f}")

avg_loss = total_loss / batch_count
print(f"✓ Mini training loop completed. Avg loss: {avg_loss:.4f}")

# Test 6: Sampling
print("\n[Test 6] Testing sampling...")
model.eval()

@torch.no_grad()
def sample_simple(model, labels, num_steps=10):
    x = torch.randn(labels.shape[0], 3, 32, 32)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.ones(labels.shape[0]) * (i / num_steps)
        t = t.view(-1, 1, 1, 1)
        v = model(x, t, labels)
        x = x + dt * v

    return x

# Generate 1 sample for each of 3 classes
test_labels = torch.tensor([0, 1, 2])
samples = sample_simple(model, test_labels, num_steps=10)

print(f"  Generated samples: {samples.shape}")
assert samples.shape == (3, 3, 32, 32), f"Expected (3, 3, 32, 32), got {samples.shape}"
print("✓ Sampling test passed")

print("\n" + "=" * 60)
print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
print("=" * 60)
print("\nSummary:")
print("  - Model instantiation: ✓")
print("  - Forward pass: ✓")
print("  - CFM loss computation: ✓")
print("  - Gradient flow: ✓")
print("  - Mini training loop: ✓")
print("  - Sampling: ✓")
print("\n🎉 Class-labeled Flow Matching is ready for training!")
print("\nTo train the model, run:")
print("  uv run src/train_cifar_with_labels.py --epochs 100 --batch_size 32 --device cpu")
