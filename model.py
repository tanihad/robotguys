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

from torchinfo import summary

import lightning as L
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint


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

def compute_iou_min(mask1, mask2):
    mask1_binary = (mask1 > 0).int()
    mask2_binary = (mask2 > 0).int()

    # Compute intersection and union
    mask1_size = torch.sum(mask1_binary, dim=(1, 2, 3))
    mask2_size = torch.sum(mask2_binary, dim=(1, 2, 3))
    intersection = torch.sum(mask1_binary & mask2_binary, dim=(1, 2, 3))
    union = torch.minimum(mask1_size, mask2_size)

    return intersection / (union + 1e-8)


class ImageMixtureDataset(Dataset):
    def __init__(self, img_dir, mask_dir, goal_dir, data_range):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.goal_dir = goal_dir
        self.data_range = data_range

        pattern = re.compile(r"img(\d+)\.png")

        self.length = -1
        for file_name in os.listdir(img_dir):
            match = pattern.match(file_name)
            if match:
                number = int(match.group(1))  # Extract the number
                self.length = max(self.length, number)
        
        assert self.length > data_range[1]
        self.length = data_range[1] - data_range[0]

        self.goal_directions = np.load(self.goal_dir)


    def __len__(self):
        return self.length

    
    def read_img_mask(self, idx):
        path_img = os.path.join(self.img_dir, f"img{idx}.png")
        path_mask = os.path.join(self.mask_dir, f"mask{idx}.png")
        image1 = rgb_to_grayscale(read_image(path_img, mode=ImageReadMode.RGB))
        mask1 = rgb_to_grayscale( read_image(path_mask, mode=ImageReadMode.RGB))

        # Convert to float and normalize to [0, 1]
        image1 = image1.float() / 255.0

        return image1, mask1, torch.from_numpy(self.goal_directions[idx]).to(torch.float16)


    def __getitem__(self, idx):
        irand = randrange(self.data_range[0], self.data_range[1])
        image1, gmix1, gdir1 = self.read_img_mask(idx)
        image2, gmix2, gdir2 = self.read_img_mask(irand)

        return image1, gmix1, gdir1, image2, gmix2, gdir2


class EmbeddingNetwork(nn.Module):
    def __init__(self, embedding_size=16):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Output: [16, 64, 64]
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # Output: [32, 32, 32]
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # Output: [64, 16, 16]
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # Output: [128, 8, 8]
        self.bn4 = nn.BatchNorm2d(128)

        self.gap = nn.AdaptiveAvgPool2d(1)  # Output: [128, 1, 1] (then flatten and concatenate with direction)
        self.fc = nn.Linear(128+2, embedding_size)  # Output: [embedding_size]

    def forward(self, x, direct):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = self.gap(x)  # [B, 128, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [B, 128]
        x = torch.cat((x, direct), 1)

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
        x, y, dir1, dir2 = x.to(device), y.to(device), dir1.to(device), dir2.to(device)
        xe = self.embedding(x, dir1)
        ye = self.embedding(y, dir2)

        sim = nn.functional.cosine_similarity(xe, ye)
        iou = 2 * compute_iou_min(mask1, mask2) - 1
        loss = nn.functional.mse_loss(sim, iou)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask1, dir1, y, mask2, dir2 = batch
        x, y = x.to(device), y.to(device)
        xe = self.embedding(x, dir1)
        ye = self.embedding(y, dir2)

        sim = nn.functional.cosine_similarity(xe, ye)
        iou = 2 * compute_iou_min(mask1, mask2) - 1
        loss = nn.functional.mse_loss(sim, iou)

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def train():
    model = EmbeddingNetwork().to(device)
    print(model)
    summary(model, input_data=[torch.randn(1, 1, 64, 64).to(device), torch.randn(1, 2).to(device)])
    model_l = EmbeddingNetworkModule(model)

    train_data = ImageMixtureDataset("imgs", "masks", "goal_directions.npy", (0, 9000))
    valid_data = ImageMixtureDataset("imgs", "masks", "goal_directions.npy", (9000, 10000-2))

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_data, batch_size=64, shuffle=True, num_workers=4)

    wandb_logger = WandbLogger(project="embedding-network-training2")
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')

    trainer = L.Trainer(max_epochs=1000, callbacks=[checkpoint_callback], logger=wandb_logger, precision=16, accelerator="gpu", devices=1)
    trainer.fit(model=model_l, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    trainer.test(model=model_l, dataloaders=valid_dataloader, ckpt_path="best")


def main():
    train()


if __name__ == '__main__':
    main()
