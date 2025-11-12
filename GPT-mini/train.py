import argparse
import torch
from tqdm import tqdm
from config import GPTConfig

from model.gpt import GPT
from data.tokenizer import Tokenizer
from data.dataloader import create_dataloaders
from utils import save_checkpoint, count_parameters

def train_epoch(model, optimizer, scheduler, loader, device):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(loader, desc='train'):
        x = x.to(device)
        y = y.to(device)
        logits, loss = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x,y in tqdm(loader, desc='eval'):
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, targets=y)
            total_loss += loss
    return  total_loss / len(loader)

def main():
    config = GPTConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config).to(device)
    count_parameters(model)

    train_loader, val_loader = create_dataloaders("/home/yu/code/GPT-mini/data/test.jsonl")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    for epoch in range(2):
        train_loss = train_epoch(model, optimizer, scheduler, train_loader, device)
        val_loss = eval_epoch(model, val_loader, device)
        print(f"Epoch {epoch}: Train {train_loss:.4f}, Val {val_loss:.4f}")
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, f"checkpoints/model_epoch_{epoch}.pt")

if __name__ == "__main__":
    main()