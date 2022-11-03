import torch
from torch.utils import data
from torchvision import datasets, transforms


class AnimeDataset:
    def __init__(self, config: dict):
        train_transform = transforms.Compose([transforms.Resize((64, 64)),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.train_dataset = datasets.ImageFolder(
            root=config["dataset_location"], transform=train_transform)
        self.train_loader = data.DataLoader(dataset=self.train_dataset, batch_size=config["batch_size"],
                                            shuffle=True, drop_last=True)
