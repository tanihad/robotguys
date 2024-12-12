import os

import torch
from torch import nn
from torch import optim, nn, utils, Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import lightning as L

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class ImageMixtureDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image1, gmix1 = read_image(img_path)

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image2, gmix2 = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        return image, label


class EmbeddingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            nn.MaxPool2d(2, 2),

            nn.LayerNorm(),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3),
            nn.MaxPool2d(2, 2),

            nn.LayerNorm(),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3),
            nn.LayerNorm(),
            nn.ReLU(),
            nn.Conv2d(3, 16, 5),
            nn.MaxPool2d(2, 2),

            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class EmbeddingNetworkModule(L.LightningModule):
    super().__init__(embedding)
        self.embedding = embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        xe = self.embedding(x)
        ye = self.embedding(y)

        sim = nn.functional.cosine_similarity(x_hat, x)
        iou = compute_iou(todo, todo2)
        loss = nn.functional.mse_loss(sim, iou)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def train():
    model = EmbeddingNetwork().to(device)
    model_l = EmbeddingNetworkModule(model)

    # TODO: train_loader
    train_data = ImageMixtureDataset()
    test_data = ImageMixtureDataset()

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


    trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(model=model_l, train_dataloaders=train_loader)

