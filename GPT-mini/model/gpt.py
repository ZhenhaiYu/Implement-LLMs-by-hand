import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import Block

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        # tansformer blocks
        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_layer)]
        )
        self.ln_final = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        # tie weights
        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx (B, T)
        B, T = idx.size() # idx: token ids of input
        # (B, T, C)
        tok_emb = self.token_embedding(idx)

        #  (1, T, C)
        pos_emb = self.pos_embedding(
            torch.arange(T, device=idx.device).unsqueeze(0)
        )

        x = tok_emb + pos_emb
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        # (B, T, V)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.view(B * T, -1)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            if top_k is not None:
                # top-k sampling
                topk_vals, topk_idx = probs.topk(top_k, dim=-1)
                probs = torch.zeros_like(probs).scatter(-1, topk_idx, topk_vals)
                probs = probs / probs.sum(-1, keepdim=True)
            next_token = torch.multionmial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx