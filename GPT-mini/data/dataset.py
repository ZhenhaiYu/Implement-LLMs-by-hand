from torch.utils.data import Dataset
import torch
import json
from .tokenizer import Tokenizer

class MyDataset(Dataset):
    def __init__(self, path, block_size=512, max_lines=1000):
        self.block_size = block_size
        self.max_lines = max_lines
        self.tokenizer = Tokenizer()

        raw_texts = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                try:
                    j = json.loads(line)
                    text = j.get('text', '')
                    if text:
                        raw_texts.append(text)
                except json.JSONDecodeError:
                    continue

        full_encoded = []
        for t in raw_texts:
            enc = self.tokenizer.encode(t)
            print(f"text:{t},encoded length:{len(enc)}")
            full_encoded.extend(enc + [self.tokenizer.enc_id])
        print(f"FULL encoded length:{len(full_encoded)}")

        self.encoded_chunks = []

        for i in range(0, len(full_encoded), self.block_size):
            chunk = full_encoded[i:i + self.block_size + 1]
            if len(chunk) < block_size + 1:
                chunk = chunk + [self.tokenizer.enc_id] * (self.block_size + 1 - len(chunk))
            self.encoded_chunks.append(chunk)

    def __len__(self):
        return len(self.encoded_chunks)

    def __getitem__(self, idx):
        chunk = self.encoded_chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y