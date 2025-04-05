# data/download_affectnethq.py

import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

def download_and_organize_affectnethq(save_dir=None):
    """
    Download AffectNetHQ dataset from Hugging Face and organize it into folders by emotion class.

    Args:
        save_dir (str): Directory where the organized dataset will be saved.
    """
    if save_dir is None:
        # Save relative to the *project's data directory* NOT where you call it from
        base_dir = os.path.dirname(os.path.abspath(__file__))  # directory where this script lives
        save_dir = os.path.join(base_dir, "affectnethq")       # inside 'data/affectnethq'

    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving dataset to: {save_dir}")

    print("Downloading AffectNetHQ dataset...")
    dataset = load_dataset("Piro17/affectnethq", split="train")

    # Correct 7-class mapping
    label_to_emotion = {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Neutral",
        5: "Sad",
        6: "Surprised"
    }

    for i in tqdm(range(len(dataset)), desc="Saving images"):
        item = dataset[i]
        img = item['image']
        label = item['label']

        if label not in label_to_emotion:
            continue

        emotion = label_to_emotion[label]
        class_dir = os.path.join(save_dir, emotion)
        os.makedirs(class_dir, exist_ok=True)

        img_path = os.path.join(class_dir, f"{i}.jpg")
        # img_path = os.path.join(class_dir, f"{emotion.lower()}_{i}.jpg")
        img.save(img_path)

    print(f"Dataset organized successfully at {save_dir}.")

if __name__ == "__main__":
    download_and_organize_affectnethq()