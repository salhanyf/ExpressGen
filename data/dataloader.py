# data/dataloader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_transforms(mode='train', img_size=256):
    """
    Returns preprocessing and augmentation transforms for the dataset.

    Args:
        mode (str): 'train', 'val', or 'test'
        img_size (int): Target image size (height, width)

    Returns:
        torchvision.transforms.Compose
    """
    base = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]

    if mode == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            *base[1:]
        ])
    return transforms.Compose(base)

def create_dataloaders(img_dir, batch_size=32, val_split=0.1, test_split=0.1, img_size=256, num_workers=4):
    """
    Create train, val, test dataloaders.

    Args:
        img_dir (str): Path to dataset
        batch_size (int): Batch size
        val_split (float): Fraction of dataset for validation
        test_split (float): Fraction of dataset for testing
        img_size (int): Image size
        num_workers (int): Number of DataLoader workers

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    dataset = datasets.ImageFolder(root=img_dir, transform=get_transforms('train', img_size))
    total = len(dataset)
    val_size = int(val_split * total)
    test_size = int(test_split * total)
    train_size = total - val_size - test_size

    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Assign non-augmented transforms to val/test
    val_data.dataset.transform = get_transforms('val', img_size)
    test_data.dataset.transform = get_transforms('test', img_size)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader