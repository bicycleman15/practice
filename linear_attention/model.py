# most basic version of linear attention

import torch
import torch.nn as nn
import einops

from dataclasses import dataclass

@dataclass
class LinearAttentionConfig:

    dim = 64

    num_heads = 1

class LinearAttention(nn.Module):
    def __init__(self, config: LinearAttentionConfig):
        super().__init__()

        self.config = config
        self.num_heads = config.num_heads
        self.dim = config.dim

        self.phi = nn.ReLU()

        self.wqkv = nn.Linear(self.dim, 3 * self.dim, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)

    def forward(self, x):
        # x: [B, L, D]
        
        q, k, v = self.wqkv(x).chunk(3, dim=-1)

        q = self.phi(q)
        k = self.phi(k)

        f = lambda x: einops.rearrange(x, "b n (h d) -> b h n d", h=self.num_heads)

        q = f(q)
        k = f(k)
        v = f(v)

        kv = einops.rearrange(k, "... n d -> ... d n") @ v # [B, H, D, N] @ [B, H, N, D] -> [B, H, D, D]

        num = q @ kv # [B, H, N, D] @ [B, H, D, D] -> [B, H, N, D]

        k_sum = k.sum(dim=-2).unsqueeze(-1) # [B, H, D, 1]
        den = q @ k_sum # [B, H, N, D] @ [B, H, D, 1] -> [B, H, N, 1]
        den = den + 1e-5

        y = num / den

        y = einops.rearrange(y, "b h n d -> b n (h d)")
        y = self.wo(y)

        return y


if __name__ == "__main__":

    config = LinearAttentionConfig()

    attention = LinearAttention(config)

    batch_size = 8
    seqlen = 128
    dim = config.dim

    x = torch.randn((batch_size, seqlen, dim))
    print(x.shape)

    x = attention(x)

    print(x.shape)