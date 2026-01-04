import torch
from unet import UNet
from models import OTFlowMatching
import argparse

def test_unet():
    print("Testing UNet...")
    model = UNet(in_channels=3, out_channels=3)
    x = torch.randn(2, 3, 32, 32)
    t = torch.rand(2)
    out = model(t, x)
    assert out.shape == x.shape
    print("UNet Forward Pass OK")

def test_fm():
    print("Testing FM...")
    fm = OTFlowMatching()
    # Mock model
    model = lambda t, x: x 
    x1 = torch.randn(10, 2)
    loss = fm.compute_loss(model, x1)
    print("FM Loss Calculation OK")
    
if __name__ == "__main__":
    test_unet()
    test_fm()
    print("All Setup Tests Passed.")
