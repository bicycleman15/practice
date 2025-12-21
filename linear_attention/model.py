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

        # [B, H, N, D, 1] @ [B, H, N, 1, D] -> [B, H, N, D, D] -> [B, H, D, D]
        kv = (k.unsqueeze(-1) * v.unsqueeze(-1).transpose(-2, -1)).cumsum(dim=-3) 
        num = q.unsqueeze(-2) @ kv # [B, H, N, 1, D] @ [B, H, N, D, D] -> [B, H, N, 1, D]
        num = num.squeeze(-2) # [B, H, N, D]

        k_sum = k.cumsum(dim=-2).unsqueeze(-1) # [B, H, N, D, 1]
        den = q.unsqueeze(-1).transpose(-2, -1) @ k_sum # [B, H, N, 1, D] @ [B, H, N, D, 1] -> [B, H, N, 1]
        den = den.squeeze(-1) + 1e-5 # [B, H, N, 1]

        y = num / den

        y = einops.rearrange(y, "b h n d -> b n (h d)")
        y = self.wo(y)

        return y

    def inference(self, x, kv, k_sum):
        # kv: [B, H, D, D]
        # k_sum: [B, H, D, 1]

        q, k, v = self.wqkv(x).chunk(3, dim=-1)

        q = self.phi(q)
        k = self.phi(k)

        f = lambda x: einops.rearrange(x, "b n (h d) -> b h n d", h=self.num_heads)

        q = f(q)
        k = f(k)
        v = f(v)

        kv = kv + (k.transpose(-2, -1) @ v) # [B, H, D, 1] @ [B, H, 1, D] # O(D^2) cost during inference
        k_sum = k_sum + k.transpose(-2, -1) # [B, H, D, 1]

        num = q @ kv # [B, H, 1, D]
        den = q @ k_sum + 1e-5 # [B, H, 1, D]

        y = num / den
        y = einops.rearrange(y, "b h n d -> b n (h d)")
        y = self.wo(y)

        # return with the new kv cache
        return y, kv, k_sum

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

    # test forward vs inference
    test_input = torch.randn((2, 16, dim))
    forward_out = attention(test_input)
    
    head_dim = dim // config.num_heads
    # define KV cache
    kv = torch.zeros((2, config.num_heads, head_dim, head_dim))
    k_sum = torch.zeros((2, config.num_heads, head_dim, 1))
    
    inference_outs = []
    for t in range(16):
        out, kv, k_sum = attention.inference(test_input[:, t:t+1, :], kv, k_sum)
        inference_outs.append(out)
    inference_out = torch.cat(inference_outs, dim=1)
    
    assert torch.allclose(forward_out, inference_out, rtol=1e-4, atol=1e-5)
    print("test passed")