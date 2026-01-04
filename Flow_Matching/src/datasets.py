import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import requests
import tarfile
from tqdm import tqdm

def download_imagenette(root):
    """Downloads and extracts Imagenette dataset."""
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    filename = "imagenette2-320.tgz"
    filepath = os.path.join(root, filename)
    dataset_dir = os.path.join(root, "imagenette2-320")
    
    if os.path.exists(dataset_dir):
        print(f"Imagenette found at {dataset_dir}")
        return dataset_dir

    os.makedirs(root, exist_ok=True)
    
    print(f"Downloading Imagenette from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, "wb") as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
            
    print("Extracting...")
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(path=root)
        
    os.remove(filepath)
    return dataset_dir

def get_8_gaussians(n_samples=10000, scale=4.0):
    centers = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1.0/np.sqrt(2), 1.0/np.sqrt(2)), (1.0/np.sqrt(2), -1.0/np.sqrt(2)),
        (-1.0/np.sqrt(2), 1.0/np.sqrt(2)), (-1.0/np.sqrt(2), -1.0/np.sqrt(2))
    ]
    centers = [(scale * x, scale * y) for x, y in centers]
    dataset = []
    for i in range(n_samples):
        point = np.random.randn(2) * 0.5
        center = centers[np.random.randint(len(centers))]
        point[0] += center[0]
        point[1] += center[1]
        dataset.append(point)
    return np.array(dataset, dtype=np.float32)

def get_dataloader(dataset_name, batch_size, num_workers=4, image_size=None, data_root='./data'):
    """
    Args:
        dataset_name: '8gaussians', 'cifar10', 'imagenet', 'celeba', 'imagenette'
        batch_size: int
        image_size: int, resize images to this size. Defaults: 32 for CIFAR, 64 for others.
        data_root: str, path to dataset storage.
    """
    if dataset_name == '8gaussians':
        data = get_8_gaussians(n_samples=50000)
        tensor_x = torch.Tensor(data)
        dataset = TensorDataset(tensor_x)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
    # Image Transforms
    if image_size is None:
        image_size = 32 if dataset_name == 'cifar10' else 64
        
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [-1, 1]
    ])
    
    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                             download=True, transform=transform)
        
    elif dataset_name == 'imagenette':
        dataset_dir = download_imagenette(data_root)
        train_dir = os.path.join(dataset_dir, 'train')
        dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)

    elif dataset_name == 'imagenet':
        # Expects standard ImageNet structure: data_root/train/class_xxx/img.jpg
        train_dir = os.path.join(data_root, 'train')
        if not os.path.exists(train_dir):
            if os.path.exists(data_root):
                 print(f"Loading ImageNet from {data_root}...")
                 train_dir = data_root
            else:
                raise FileNotFoundError(f"ImageNet data not found at {data_root}. Please ensure 'train' folder exists or use 'imagenette'.")
        
        dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
        
    elif dataset_name == 'celeba':
        # CelebA is often used for 64x64 generation
        dataset = torchvision.datasets.CelebA(root=data_root, split='train', 
                                            download=True, transform=transform)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                      num_workers=num_workers, pin_memory=True)

