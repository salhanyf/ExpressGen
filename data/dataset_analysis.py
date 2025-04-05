# data/dataset_analysis.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets

def load_dataset(root_dir, transform=None):
    """
    Load a dataset from a directory.

    Args:
        root_dir (str): Path to the dataset directory.
        transform (torchvision.transforms, optional): Transformations to apply.

    Returns:
        torchvision.datasets.ImageFolder: Loaded dataset.
    """
    dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    return dataset

def get_dataset_stats(dataset):
    """
    Print basic statistics about the dataset.

    Args:
        dataset (torchvision.datasets.ImageFolder): Loaded dataset.
    """
    print(f"Number of samples: {len(dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"Classes: {dataset.classes}")

def plot_class_distribution(dataset):
    """
    Plot the distribution of classes in the dataset.

    Args:
        dataset (torchvision.datasets.ImageFolder): Loaded dataset.
    """
    labels = [label for _, label in dataset]
    labels = torch.tensor(labels)

    class_counts = torch.bincount(labels)
    class_names = dataset.classes

    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_counts.numpy())
    plt.xlabel('Classes')
    plt.ylabel('Number of images')
    plt.title('Class Distribution in AffectNetHQ')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

def show_samples_per_class(dataset, samples_per_class=4):
    """
    Display a fixed number of sample images per class.

    Args:
        dataset (torchvision.datasets.ImageFolder): Loaded dataset.
        samples_per_class (int, optional): Number of samples per class to display. Default is 4.
    """
    class_to_indices = {cls_idx: [] for cls_idx in range(len(dataset.classes))}

    for idx, (_, label) in enumerate(dataset):
        if len(class_to_indices[label]) < samples_per_class:
            class_to_indices[label].append(idx)

    total_classes = len(dataset.classes)
    plt.figure(figsize=(samples_per_class * 2, total_classes * 2))

    for class_idx, indices in class_to_indices.items():
        for i, idx in enumerate(indices):
            img, label = dataset[idx]
            img = img.permute(1, 2, 0).numpy()

            plt_idx = class_idx * samples_per_class + i + 1
            plt.subplot(total_classes, samples_per_class, plt_idx)
            plt.imshow(img)
            plt.title(dataset.classes[label])
            plt.axis('off')

    plt.tight_layout()
    plt.show()