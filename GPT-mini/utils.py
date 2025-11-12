import os
import torch

def save_checkpoint(state,path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
