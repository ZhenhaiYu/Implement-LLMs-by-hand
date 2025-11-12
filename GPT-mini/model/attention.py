import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    含 causal mask
    采用一次性 qkv投影以加速计算
    """
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        assert  self.head_size * self.n_head == self.n_embd

        self.qkv = nn.Linear(self.n_embd, self.n_embd * 3)
        self.out = nn.Linear(self.n_embd, self.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size)).unsqueeze(0)
        mask = mask.view(1, 1, config.block_size, config.block_size)
        self.register_buffer('causal_mask', mask)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x) # (B, T, 3*C)
        qkv= qkv.view(B, T, 3, self.n_head, self.head_size).permute(2, 0, 3, 1, 4)
        # qkv (3,B, n_head, head_size)
        q, k ,v = qkv[0], qkv[1], qkv[2]

        # (B,n_head, T, T)
        att = (q @ k.transpose(-2,-1)) /math.sqrt(self.head_size)

        # apply causal mask
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0,float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v
        out = out.transpose(1,2).contiguous().view(B, T, C)
        out = self.out(out)
        out= self.dropout(out)
        return  out