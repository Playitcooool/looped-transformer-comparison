"""Causal pre-norm GPT blocks, optionally reused across depth."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8192
    seq_len: int = 256
    width: int = 512
    heads: int = 8
    depth: int = 12
    loop_layers: int = 3
    loops: int = 4
    dropout: float = 0.0

    def __post_init__(self):
        for name in ('vocab_size', 'seq_len', 'width', 'heads', 'depth', 'loop_layers', 'loops'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.width % self.heads:
            raise ValueError('width must be divisible by heads')
        if self.loop_layers * self.loops != self.depth:
            raise ValueError('loop_layers * loops must equal depth')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1)')


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.dropout = c.heads, c.dropout
        self.ln1, self.ln2 = nn.LayerNorm(c.width), nn.LayerNorm(c.width)
        self.qkv = nn.Linear(c.width, 3 * c.width)
        self.proj = nn.Linear(c.width, c.width)
        self.mlp = nn.Sequential(nn.Linear(c.width, 4 * c.width), nn.GELU(), nn.Linear(4 * c.width, c.width))
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q, k, v = [a.view(b, t, self.heads, d // self.heads).transpose(1, 2) for a in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                          dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.proj(y.transpose(1, 2).contiguous().view(b, t, d)))
        return x + self.drop(self.mlp(self.ln2(x)))


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig, architecture: str):
        super().__init__()
        if architecture not in ('standard', 'looped'):
            raise ValueError('architecture must be standard or looped')
        rng = torch.get_rng_state()
        self.config, self.architecture = config, architecture
        self.token = nn.Embedding(config.vocab_size, config.width)
        self.position = nn.Embedding(config.seq_len, config.width)
        self.blocks = nn.ModuleList([Block(config) for _ in range(
            config.depth if architecture == 'standard' else config.loop_layers)])
        self.norm = nn.LayerNorm(config.width)
        torch.set_rng_state(rng)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens):
        if tokens.ndim != 2 or not 0 < tokens.size(1) <= self.config.seq_len:
            raise ValueError('expected [batch, time] within configured context length')
        x = self.token(tokens) + self.position(torch.arange(tokens.size(1), device=tokens.device))
        for _ in range(self.config.loops if self.architecture == 'looped' else 1):
            for block in self.blocks:
                x = block(x)
        return F.linear(self.norm(x), self.token.weight)
