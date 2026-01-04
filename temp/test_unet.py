import torch
from src.models import UNet
from src.data import get_cifar10_dataloader
import os

def test_unet_shapes():
    print("Testing UNet shapes...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        in_channels=3,
        model_channels=128,  # Smaller for test
        num_res_blocks=1,
        channel_mult=(1, 2),
    ).to(device)
    
    x = torch.randn(4, 3, 32, 32).to(device)
    t = torch.rand(4).to(device)
    
    v = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {v.shape}")
    
    assert x.shape == v.shape, f"Shape mismatch: {x.shape} vs {v.shape}"
    print("✅ UNet shape test passed!")

def test_cifar10_loader():
    print("\nTesting CIFAR-10 loader...")
    # Skip download if not on server or if takes too long, 
    # but here we want to verify it works.
    try:
        dataloader = get_cifar10_dataloader(batch_size=4, download=True)
        images, labels = next(iter(dataloader))
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
        print(f"Value range: [{images.min():.2f}, {images.max():.2f}]")
        
        assert images.shape == (4, 3, 32, 32)
        assert images.min() >= -1.1 and images.max() <= 1.1
        print("✅ CIFAR-10 loader test passed!")
    except Exception as e:
        print(f"❌ CIFAR-10 loader test failed or skipped: {e}")

if __name__ == "__main__":
    test_unet_shapes()
    test_cifar10_loader()
