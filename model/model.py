# model/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A simple residual block with two 3x3 convolutions and instance normalization.
    """
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim, affine=False),
        )

    def forward(self, x):
        return x + self.block(x)


class SelfAttention(nn.Module):
    """
    Lightweight self-attention layer (from SAGAN) for capturing long-range dependencies.
    """
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key   = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(in_dim, in_dim,     kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query(x).view(B, -1, H*W).permute(0, 2, 1)  # B x N x C'
        proj_key   = self.key(x).view(B, -1, H*W)                     # B x C' x N
        energy     = torch.bmm(proj_query, proj_key)                 # B x N x N
        attention  = F.softmax(energy, dim=-1)

        proj_value = self.value(x).view(B, -1, H*W)                  # B x C x N
        out        = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x N
        out        = out.view(B, C, H, W)

        return self.gamma * out + x


class Generator(nn.Module):
    """
    Expression-transform generator with residual blocks and self-attention.
    """
    def __init__(self, img_channels=3, num_classes=7, img_size=160):
        super().__init__()
        self.img_size = img_size
        self.label_emb = nn.Embedding(num_classes, num_classes)

        # Encoder (downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv2d(img_channels + num_classes, 64, 4, 2, 1),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Bottleneck
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(4)]
        )
        self.attn       = SelfAttention(256)

        # Decoder (upsampling)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256 + num_classes, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x, labels):
        # Embed and expand labels to spatial maps
        label_embed = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        label_map   = label_embed.expand(-1, -1, x.size(2), x.size(3))

        # Encode
        x = torch.cat([x, label_map], dim=1)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        # Residual + Attention
        x = self.res_blocks(x)
        x = self.attn(x)

        # Inject labels for decoder
        label_map_mid = label_embed.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, label_map_mid], dim=1)

        # Decode
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        return x


class Discriminator(nn.Module):
    """
    PatchGAN-style discriminator with self-attention.
    """
    def __init__(self, img_channels=3, num_classes=7):
        super().__init__()
        # convolutional backbone
        self.main = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ← this line was missing
        self.attn     = SelfAttention(256)

        # heads
        self.adv_head = nn.Conv2d(256, 1,          kernel_size=1)
        self.cls_head = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        h = self.main(x)
        # ← apply attention here
        h = self.attn(h)
        h = F.adaptive_avg_pool2d(h, 1)  # squeeze to (B, C, 1, 1)
        out_adv = self.adv_head(h).view(h.size(0))
        out_cls = self.cls_head(h).view(h.size(0), -1)
        return out_adv, out_cls
