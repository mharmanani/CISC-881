import torch
from torch.utils.data import Dataset

def SegmentationDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, mask = self.dataset[idx]

        return image, mask

