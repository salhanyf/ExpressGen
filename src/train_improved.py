# src/train_improved.py

import os
import time
import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision.models import vgg19

def compute_class_weights(train_loader: DataLoader, device: torch.device) -> torch.Tensor:
    """
    Compute weights inversely proportional to class frequency in the training set.
    """
    all_labels = [label for _, label in train_loader.dataset]
    counts = Counter(all_labels)
    total = sum(counts.values())
    weights = torch.tensor(
        [total / counts[i] for i in range(len(counts))],
        dtype=torch.float32,
        device=device
    )
    return weights


@torch.no_grad()
def evaluate(
    G: nn.Module,
    D: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor
) -> tuple[float, float]:
    """
    Evaluate generator and discriminator losses on a validation set.
    """
    G.eval()
    D.eval()
    adv_loss_fn = nn.BCEWithLogitsLoss()
    cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    num_classes = class_weights.size(0)

    d_losses, g_losses = [], []
    for real_images, real_labels in dataloader:
        real_images = real_images.to(device)
        real_labels = real_labels.to(device)
        bsz = real_images.size(0)

        # sample random targets ≠ real labels
        rand_labels = torch.randint(0, num_classes, (bsz,), device=device)
        mask = rand_labels == real_labels
        rand_labels[mask] = (rand_labels[mask] + 1) % num_classes

        # Discriminator forward
        out_adv_real, out_cls_real = D(real_images)
        d_adv_real = adv_loss_fn(out_adv_real, torch.ones_like(out_adv_real))
        d_cls_real = cls_loss_fn(out_cls_real, real_labels)

        fake_images = G(real_images, rand_labels)
        out_adv_fake, _ = D(fake_images)
        d_adv_fake = adv_loss_fn(out_adv_fake, torch.zeros_like(out_adv_fake))

        d_losses.append((d_adv_real + d_adv_fake + d_cls_real).item())

        # Generator forward
        out_adv_fake, out_cls_fake = D(fake_images)
        g_adv = adv_loss_fn(out_adv_fake, torch.ones_like(out_adv_fake))
        g_cls = cls_loss_fn(out_cls_fake, rand_labels)
        g_losses.append((g_adv + g_cls).item())

    return sum(d_losses) / len(d_losses), sum(g_losses) / len(g_losses)


def train(
    G: nn.Module,
    D: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 20,
    save_dir: str = "../results"
) -> None:
    """
    Train Generator and Discriminator with reconstruction and perceptual losses.
    """
    # set up perceptual feature extractor (VGG19)
    perc_model = vgg19(pretrained=True).features[:16].eval().to(device)
    for p in perc_model.parameters():
        p.requires_grad = False

    os.makedirs(save_dir, exist_ok=True)
    start_time = time.time()

    # losses and weights
    class_weights = compute_class_weights(train_loader, device)
    adv_loss_fn   = nn.BCEWithLogitsLoss()
    cls_loss_fn   = nn.CrossEntropyLoss(weight=class_weights)
    num_classes   = class_weights.size(0)
    lambda_rec    = 10.0
    lambda_perc   =  5.0

    # optimizers
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    G.train()
    D.train()
    for epoch in range(1, num_epochs + 1):
        total_batches = len(train_loader)
        running_d, running_g = 0.0, 0.0

        for i, (real_images, real_labels) in enumerate(train_loader, start=1):
            real_images = real_images.to(device)
            real_labels = real_labels.to(device)
            bsz = real_images.size(0)

            # sample random target labels
            rand_labels = torch.randint(0, num_classes, (bsz,), device=device)
            mask = rand_labels == real_labels
            rand_labels[mask] = (rand_labels[mask] + 1) % num_classes

            # Discriminator update
            out_adv_real, out_cls_real = D(real_images)
            d_adv_real = adv_loss_fn(out_adv_real, torch.ones_like(out_adv_real))
            d_cls_real = cls_loss_fn(out_cls_real, real_labels)
            fake_images = G(real_images, rand_labels)
            out_adv_fake, _ = D(fake_images.detach())
            d_adv_fake = adv_loss_fn(out_adv_fake, torch.zeros_like(out_adv_fake))
            d_loss = d_adv_real + d_adv_fake + d_cls_real
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            # Generator update
            fake_images = G(real_images, rand_labels)
            out_adv_fake, out_cls_fake = D(fake_images)
            g_adv       = adv_loss_fn(out_adv_fake, torch.ones_like(out_adv_fake))
            g_cls       = cls_loss_fn(out_cls_fake, rand_labels)
            recon_loss  = F.l1_loss(fake_images, real_images)
            feat_real   = perc_model(real_images)
            feat_fake   = perc_model(fake_images)
            perc_loss   = F.l1_loss(feat_fake, feat_real)
            g_loss      = g_adv + g_cls + lambda_rec * recon_loss + lambda_perc * perc_loss
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()

            running_d += d_loss.item()
            running_g += g_loss.item()
            if i % 10 == 0 or i == total_batches:
                pct = i / total_batches * 100
                print(f"\rEpoch {epoch}/{num_epochs} [{pct:.1f}%]", end="")

        # end epoch metrics
        avg_d = running_d / total_batches
        avg_g = running_g / total_batches
        val_d, val_g = evaluate(G, D, val_loader, device, class_weights)
        elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        print(f"\nEpoch {epoch} | Time {elapsed} | Train_D {avg_d:.4f} | Train_G {avg_g:.4f} | Val_D {val_d:.4f} | Val_G {val_g:.4f}")

        # save checkpoints
        torch.save(G.state_dict(), os.path.join(save_dir, f"G_epoch{epoch}.pth"))
        torch.save(D.state_dict(), os.path.join(save_dir, f"D_epoch{epoch}.pth"))

        # visualize sample every 5 epochs
        if epoch % 5 == 0:
            G.eval()
            imgs, lbls = next(iter(val_loader))
            inp    = imgs[0].unsqueeze(0).to(device)
            orig   = lbls[0].to(device)
            tgt    = torch.randint(0, num_classes, (1,), device=device)
            if tgt == orig:
                tgt = (tgt + 1) % num_classes
                
            with torch.no_grad(): 
                gen = G(inp, tgt)

            # drop batch dimension for make_grid
            real_img = (inp[0] + 1) / 2               # [3,H,W]
            gen_img  = (gen[0] + 1) / 2               # [3,H,W]
            comp     = make_grid([real_img, gen_img], nrow=2)
            
            plt.figure(figsize=(4,2)); 
            plt.imshow(comp.permute(1,2,0).cpu().numpy()); 
            plt.axis('off'); 
            plt.show()
            G.train()
