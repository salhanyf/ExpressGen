# data/dataset_analysis.py

import torch
import matplotlib.pyplot as plt
import numpy as np

def get_dataset_stats(dataset):
    """
    Print basic statistics about the dataset.

    Args:
        dataset (Dataset): A dataset object.
    """
    print(f"Number of samples: {len(dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"Classes: {dataset.classes}")
    print("")

def plot_class_distribution(dataset):
    """
    Plot the distribution of classes in the dataset.

    Args:
        dataset (Dataset): A dataset object.
    """
    #labels = [label for _, label in dataset]
    #labels = torch.tensor(labels)
    labels = torch.tensor(dataset.targets)
    class_counts = torch.bincount(labels)
    class_names = dataset.classes

    plt.figure(figsize=(8, 4))
    plt.bar(class_names, class_counts.numpy())
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution in AffectNetHQ')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

def show_samples_per_class(dataset, samples_per_class=4):
    """
    Show a grid of sample images per class.

    Args:
        dataset (Dataset): A dataset object.
        samples_per_class (int): Number of samples to show per class.
    """
    class_to_indices = {cls_idx: [] for cls_idx in range(len(dataset.classes))}

    for idx, (_, label) in enumerate(dataset):
        if len(class_to_indices[label]) < samples_per_class:
            class_to_indices[label].append(idx)

    total_classes = len(dataset.classes)
    plt.figure(figsize=(samples_per_class * 1.5, total_classes * 1.5))

    for class_idx, indices in class_to_indices.items():
        for i, idx in enumerate(indices):
            img, label = dataset[idx]
            img = img.permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)
            
            plt_idx = class_idx * samples_per_class + i + 1
            plt.subplot(total_classes, samples_per_class, plt_idx)
            plt.imshow(img)
            plt.title(dataset.classes[label])
            plt.axis('off')

    plt.tight_layout()
    plt.show()