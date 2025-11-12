from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 512 # max_seq_len
    batch_size: int = 12
    n_layer: int = 6
    n_head: int  = 12
    n_embd: int = 768 # hidden_size, hidden_dim. same as embed_dim.
    dropout: float = 0.1
    vocab_size: int = 50527 # tiktoken is used to GPT-2
    lr: float = 3e-4
    max_eopchs: int = 100
    warmup_steps: int = 0

    @property
    def head_size(self):
        return self.n_embd // self.n_head