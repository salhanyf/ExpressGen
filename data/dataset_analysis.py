import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image
from data.dataset_loader import AffectNetHQDataset  # Assuming the dataset loader is in data/

def print_dataset_info(root_dir):
    """
    Prints dataset statistics such as total images, class distribution, and image size.
    
    Args:
        root_dir (str): Path to the dataset directory.
    """
    dataset = AffectNetHQDataset(root_dir)

    num_samples = len(dataset)
    class_counts = {}

    for _, label in dataset:
        class_counts[label] = class_counts.get(label, 0) + 1

    print("📊 Dataset Information:")
    print(f"   - Total Images: {num_samples}")
    print(f"   - Number of Classes: {len(class_counts)}")
    print("   - Class Distribution:")
    for class_label, count in class_counts.items():
        print(f"     - Class {class_label}: {count} images")

def display_sample_images(root_dir, num_samples=6):
    """
    Displays a sample of images from the dataset.
    
    Args:
        root_dir (str): Path to the dataset directory.
        num_samples (int): Number of sample images to display.
    """
    dataset = AffectNetHQDataset(root_dir, transform=transforms.ToTensor())

    fig, ax = plt.subplots(figsize=(10, 5))
    images, _ = zip(*[dataset[i] for i in range(num_samples)])
    
    # Create a grid of images
    grid = make_grid(images, nrow=num_samples, padding=2, normalize=True)
    ax.imshow(grid.permute(1, 2, 0))
    ax.axis("off")
    ax.set_title("Sample Images from Dataset")
    plt.show()