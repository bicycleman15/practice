import torch
import torch.nn as nn
import einops
import math

from dataclasses import dataclass

@dataclass
class Config:

    batch_size = 8
    block_size = 512

    dim = 128

    num_heads = 2

    def __init__(self):
        assert self.dim % self.num_heads == 0
        self.head_dim = self.dim // self.num_heads


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.dim = config.dim
        self.head_dim = config.head_dim
        self.num_heads = config.num_heads

    def forward(self, q, k, v, attn_mask=None, is_causal=True):
        
        B, N, _ = q.shape

        q, k, v = map(lambda x: einops.rearrange(x, "b n (h d) -> b h n d", h=self.num_heads, d=self.head_dim), (q, k, v))

        # q: [B, H, N, D]

        scale = 1 / math.sqrt(self.head_dim)
        scores = q @ k.transpose(-1, -2) * scale

        # apply mask
        scores = torch.masked_fill(scores, ~attn_mask.view(1, 1, N, N), -float("inf")) # [B, H, N, N]
        probs = torch.softmax(scores, dim=-1)

        y = probs @ v # [B, H, N, D]

        y = einops.rearrange(y, "b h n d -> b n (h d)", h=self.num_heads, d=self.head_dim)

        return y


def create_qkv(config):

    B, N, H, D = config.batch_size, config.block_size, config.num_heads, config.head_dim

    q = torch.randn((B, N, H*D))
    k = torch.randn((B, N, H*D))
    v = torch.randn((B, N, H*D))

    return q, k, v


def main():
    config = Config()

    q, k, v = create_qkv(config)

    attn_ref = nn.MultiheadAttention(
        embed_dim=config.dim,
        num_heads=config.num_heads,
        bias=False,
        batch_first=True
    )
    attn = Attention(config)

    attn_mask = ~torch.triu(torch.ones((config.block_size, config.block_size), dtype=torch.bool), diagonal=1)

    # y_ref = attn_ref(q, k, v, is_causal=True, attn_mask=attn_mask, need_weights=False)[0]
    # print(y_ref.shape)

    # Reshape for scaled_dot_product_attention: [B, N, H*D] -> [B, H, N, D]
    q_ref = einops.rearrange(q, "b n (h d) -> b h n d", h=config.num_heads, d=config.head_dim)
    k_ref = einops.rearrange(k, "b n (h d) -> b h n d", h=config.num_heads, d=config.head_dim)
    v_ref = einops.rearrange(v, "b n (h d) -> b h n d", h=config.num_heads, d=config.head_dim)

    y_ref = torch.nn.functional.scaled_dot_product_attention(
        q_ref, k_ref, v_ref, attn_mask=attn_mask, dropout_p=0.0, scale=(1/math.sqrt(config.head_dim)), is_causal=True #, attn_mask=mask
    )
    # Reshape back: [B, H, N, D] -> [B, N, H*D]
    y_ref = einops.rearrange(y_ref, "b h n d -> b n (h d)", h=config.num_heads, d=config.head_dim)

    y = attn(q, k, v, attn_mask=attn_mask)
    print(y.shape)

    # print(y_ref[:1, :10, :10])
    # print()
    # print(y[:1, :10, :10])

    assert torch.allclose(y_ref, y, atol=1e-6)


if __name__ == "__main__":
    main()