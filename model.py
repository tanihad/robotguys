import os
from random import randrange
import re

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim, nn, utils, Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import rgb_to_grayscale

from torchsummary import summary

import lightning as L
from lightning.pytorch.loggers.wandb import WandbLogger

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


def compute_iou(mask1, mask2):
    mask1_binary = (mask1 > 0).int()
    mask2_binary = (mask2 > 0).int()

    # Compute intersection and union
    intersection = torch.sum(mask1_binary & mask2_binary, dim=(1, 2, 3))
    union = torch.sum(mask1_binary | mask2_binary, dim=(1, 2, 3))

    return intersection / (union + 1e-8)


class ImageMixtureDataset(Dataset):
    def __init__(self, img_dir, mask_dir, goal_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.goal_dir = goal_dir

        pattern = re.compile(r"img(\d+)\.png")

        self.length = -1
        for file_name in os.listdir(img_dir):
            match = pattern.match(file_name)
            if match:
                number = int(match.group(1))  # Extract the number
                self.length = max(self.length, number)

        self.goal_directions = np.load(self.goal_dir)

    def __len__(self):
        return self.length

    def read_img_mask(self, idx):
        path_img = os.path.join(self.img_dir, f"img{idx}.png")
        path_mask = os.path.join(self.mask_dir, f"mask{idx}.png")
        image1 = rgb_to_grayscale(read_image(path_img, mode=ImageReadMode.RGB))
        mask1 = rgb_to_grayscale(read_image(path_mask, mode=ImageReadMode.RGB))

        # Convert to float and normalize to [0, 1]
        image1 = image1.float() / 255.0

        return image1, mask1, self.goal_directions[idx]

    def __getitem__(self, idx):
        irand = randrange(0, 10)
        image1, gmix1, gdir1 = self.read_img_mask(idx)
        image2, gmix2, gdir2 = self.read_img_mask(irand)

        return image1, gmix1, gdir1, image2, gmix2, gdir2


class EmbeddingNetwork(nn.Module):
    def __init__(self, embedding_size=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Output: [16, 64, 64]
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # Output: [32, 32, 32]
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # Output: [64, 16, 16]
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # Output: [128, 8, 8]
        self.bn4 = nn.BatchNorm2d(128)

        self.gap = nn.AdaptiveAvgPool2d(1)  # Output: [128, 1, 1]
        self.fc = nn.Linear(128, embedding_size)  # Output: [embedding_size]

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = self.gap(x)  # [B, 128, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [B, 128]

        x = self.fc(x)
        return x


class EmbeddingNetworkModule(L.LightningModule):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, mask1, dir1, y, mask2, dir2 = batch
        x, y = x.to(device), y.to(device)
        xe = self.embedding(x)
        ye = self.embedding(y)

        sim = nn.functional.cosine_similarity(xe, ye)
        iou = compute_iou(mask1, mask2)
        loss = nn.functional.mse_loss(sim, iou)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def train():
    model = EmbeddingNetwork().to(device)
    print(model)
    summary(model, input_size=(1, 64, 64))
    model_l = EmbeddingNetworkModule(model)

    train_data = ImageMixtureDataset("imgs", "masks", "goal_directions.npy")
    # test_data = ImageMixtureDataset("imgs", "masks")

    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=4)

    wandb_logger = WandbLogger(project="embedding-network-training")
    trainer = L.Trainer(max_epochs=10, logger=wandb_logger, precision=16, accelerator="gpu", devices=1)
    trainer.fit(model=model_l, train_dataloaders=train_dataloader)


def main():
    train()


if __name__ == '__main__':
    main()