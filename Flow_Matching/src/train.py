import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm

from models import VectorFieldNet, OTFlowMatching, VPFlowMatching, TargetFlowMatching
from unet import UNet
from datasets import get_dataloader

def plot_samples(samples, step, save_dir, dataset_name):
    """Visualization wrapper."""
    if dataset_name == '8gaussians':
        plt.figure(figsize=(6, 6))
        # Plot generated samples
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10, c='red', label='Generated')
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.title(f'Step {step}')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'step_{step:05d}.png'))
        plt.close()
    else: # Image datasets
        # Samples: [B, 3, H, W] -> needs denormalize
        # [-1, 1] -> [0, 1]
        samples = (samples + 1) / 2
        samples = samples.clamp(0, 1)
        
        # Make grid
        import torchvision.utils as vutils
        grid = vutils.make_grid(samples[:64], nrow=8, padding=2)
        vutils.save_image(grid, os.path.join(save_dir, f'step_{step:05d}.png'))

def get_fm_loss_wrapper(fm_type):
    if fm_type == 'ot':
        return OTFlowMatching(sigma_min=1e-4)
    elif fm_type == 'vp':
        return VPFlowMatching()
    elif fm_type == 'target':
        return TargetFlowMatching()
    else:
        raise ValueError(f"Unknown FM type: {fm_type}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='8gaussians', choices=['8gaussians', 'cifar10', 'imagenet', 'celeba', 'imagenette'])
    parser.add_argument('--fm_type', type=str, default='ot', choices=['ot', 'vp', 'target'])
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'unet'])
    parser.add_argument('--solver', type=str, default='euler', choices=['euler', 'dopri5'])
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log_interval', type=int, default=500)
    parser.add_argument('--save_dir', type=str, default='../results') # Save to root/results
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--image_size', type=int, default=None, help='Resolution of images (e.g. 32, 64). Default depends on dataset.')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Defaults
    if args.image_size is None:
        if args.dataset == 'cifar10':
            args.image_size = 32
        elif args.dataset in ['imagenet', 'celeba']:
            args.image_size = 64
        else:
            args.image_size = 32 # Irrelevant for 8gaussians
            
    print(f"Dataset: {args.dataset}, Image Size: {args.image_size}")
    
    # Paths
    run_name = f"{args.dataset}_{args.fm_type}_{args.model}_{args.image_size}"
    save_path = os.path.join(args.save_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    
    # Data
    dataloader = get_dataloader(args.dataset, args.batch_size, image_size=args.image_size, data_root=args.data_root)
    data_iter = iter(dataloader)
    
    # Model
    if args.model == 'mlp':
        if args.dataset != '8gaussians':
            print("Warning: MLP is likely too weak for Images. Suggest using --model unet")
            input_dim = 3 * args.image_size * args.image_size
        else:
            input_dim = 2
        model = VectorFieldNet(output_dim=input_dim, hidden_dim=128).to(device)
        
    elif args.model == 'unet':
        if args.dataset == '8gaussians':
             raise ValueError("UNet not suitable for 2D toy data.")
        # Adjust base channels for larger resolution if needed
        base_ch = 64
        if args.image_size >= 128:
            base_ch = 128
        model = UNet(in_channels=3, out_channels=3, base_channels=base_ch).to(device)
    
    # Optimization
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    fm_wrapper = get_fm_loss_wrapper(args.fm_type)
    
    print(f"Start training {run_name} for {args.iterations} iterations...")
    
    loss_history = []
    
    for step in tqdm(range(1, args.iterations + 1)):
        try:
            x1 = next(data_iter)
        except (StopIteration, ValueError): 
            data_iter = iter(dataloader)
            x1 = next(data_iter)
        
        # Handle tuple/list from dataloader
        if isinstance(x1, (list, tuple)):
            x1 = x1[0]
        
        x1 = x1.to(device)
        
        # Preprocessing for MLP on Images
        if args.model == 'mlp' and args.dataset != '8gaussians':
            x1 = x1.view(x1.size(0), -1)
            
        optimizer.zero_grad()
        loss = fm_wrapper.compute_loss(model, x1)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if step % args.log_interval == 0:
            # Sampling
            if args.dataset == '8gaussians':
                sample_shape = (2000, 2)
            else:
                sample_shape = (64, 3, args.image_size, args.image_size)
                
            samples = fm_wrapper.sample_ode(model, sample_shape, steps=100, device=device, solver=args.solver)
            
            # Post-processing for visualization
            if args.model == 'mlp' and args.dataset != '8gaussians':
                samples = samples.view(-1, 3, args.image_size, args.image_size)
                
            plot_samples(samples.cpu(), step, save_path, args.dataset)
            
            # Save Checkpoint
            torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))
            
    # Save Loss Curve
    plt.figure()
    plt.plot(loss_history)
    plt.title("Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(save_path, "loss.png"))
    
    print("Training Finished.")

if __name__ == "__main__":
    main()
