# src/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.model import Generator, Discriminator
from data.dataloader import get_dataloader
from torchvision.utils import save_image

class TrainConfig:
    def __init__(self,
                 data_dir='../data/affectnethq',
                 save_dir='../results',
                 pretrained_path='../model/pretrained/starganv2_pretrained.pth',
                 num_epochs=50,
                 batch_size=16,
                 learning_rate=1e-4,
                 img_size=256,
                 num_domains=7,
                 device=None):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.pretrained_path = pretrained_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.img_size = img_size
        self.num_domains = num_domains
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Load Dataloader
    train_loader = get_dataloader(args.data_dir, batch_size=args.batch_size, img_size=args.img_size)

    # Initialize Models
    G = Generator(img_size=args.img_size, style_dim=64, w_hpf=1).to(args.device)
    D = Discriminator(img_size=args.img_size, num_domains=args.num_domains).to(args.device)

    # Load pre-trained Generator weights
    checkpoint = torch.load(args.pretrained_path)
    G.load_state_dict(checkpoint['generator'], strict=False)
    print("✅ Loaded pre-trained Generator weights successfully!")

    # Initialize Optimizers
    g_optimizer = optim.Adam(G.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.num_epochs):
        G.train()
        D.train()
        
        loop = tqdm(train_loader, leave=True)
        for i, (real_images, labels) in enumerate(loop):
            real_images = real_images.to(args.device)
            labels = labels.to(args.device)

            # === Train Discriminator ===
            fake_labels = torch.randint(0, args.num_domains, (real_images.size(0),), device=args.device)
            style_vectors = torch.randn(real_images.size(0), 64, device=args.device)

            # Generate fake images
            fake_images = G(real_images, style_vectors)

            real_validity = D(real_images, labels)
            fake_validity = D(fake_images.detach(), fake_labels)

            d_loss_real = criterion(real_validity, torch.ones_like(real_validity))
            d_loss_fake = criterion(fake_validity, torch.zeros_like(fake_validity))
            d_loss = (d_loss_real + d_loss_fake) / 2

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # === Train Generator ===
            fake_validity = D(fake_images, fake_labels)
            g_loss_adv = criterion(fake_validity, torch.ones_like(fake_validity))

            g_optimizer.zero_grad()
            g_loss_adv.backward()
            g_optimizer.step()

            # Update tqdm loop
            loop.set_description(f"Epoch [{epoch+1}/{args.num_epochs}]")
            loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss_adv.item())

        # Save generated samples every few epochs
        if (epoch + 1) % 5 == 0:
            G.eval()
            with torch.no_grad():
                style_vectors = torch.randn(4, 64, device=args.device)
                sample_images = G(real_images[:4], style_vectors)
                save_image(sample_images, os.path.join(args.save_dir, f'epoch_{epoch+1}.png'), nrow=2, normalize=True)

    print("🎉 Training finished!")

if __name__ == "__main__":
    args = TrainConfig()
    train(args)