import torch.utils.data
from torch.utils.data import DataLoader
from .dataset import MyDataset


def create_dataloaders(path, batch_size = 12, split_ratio = 0.1):
    dataset = MyDataset(path)
    n = len(dataset)
    val_len = int(n * split_ratio)
    train_len = n - val_len
    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader