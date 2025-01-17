# custom_dataset.py
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, images, captions):
        self.images = images
        self.captions = captions

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.captions[idx]
