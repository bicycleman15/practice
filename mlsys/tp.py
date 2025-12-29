from dataclasses import dataclass

import math

import torch
import torch.nn as nn
import torch.distributed as dist

import einops

@dataclass
class ConfigTP:
    num_processes: int = 2

    dim: int = 256
    head_dim: int = 64

    def __init__(self):
        assert self.dim % self.num_processes == 0
        self.dim_reduced = self.dim // self.num_processes


class AllReduceFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # In forward pass, perform all-reduce (sum across all processes)
        # For now, just return x (actual implementation would use dist.all_reduce)
        torch.distributed.all_reduce(x, op=dist.ReduceOp.SUM)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        # In backward pass, just pass through the gradient
        return grad_output


class IdentityFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # In forward pass, just return x (identity)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        # In backward pass, perform all-reduce (sum gradients across all processes)
        torch.distributed.all_reduce(grad_output, op=dist.ReduceOp.SUM)
        return grad_output
        

def f(x):
    """Identity in forward, all-reduce in backward"""
    return IdentityFunc.apply(x)


def g(x):
    """All-reduce in forward, identity in backward"""
    return AllReduceFunc.apply(x)


class TPFeedForward(nn.Module):
    def __init__(self, config: ConfigTP):
        super().__init__()

        self.config = config
        self.num_processes = config.num_processes
        self.dim = config.dim
        self.dim_reduced = self.dim // self.num_processes

        self.w1 = nn.Linear(self.dim, self.dim_reduced)
        self.w2 = nn.Linear(self.dim_reduced, self.dim)

        self.act_fn = nn.GeLU()

    def forward(self, x):

        x = f(x)
        
        # assume x is everywhere
        x = self.w1(x) # [B, D_reduced]
        x = self.act_fn(x)
        x = self.w2(x)

        # all reduce x here
        x = g(x)

        return x


class Attention(nn.Module):

    def __init__(self, config: ConfigTP):

        self.config = config
        self.dim = config.dim
        self.dim_reduced = config.dim_reduced
        self.head_dim = config.head_dim

        # [D, D_red]
        self.wq = nn.Linear(self.dim, self.dim_reduced, bias=False)
        self.wk = nn.Linear(self.dim, self.dim_reduced, bias=False)
        self.wv = nn.Linear(self.dim, self.dim_reduced, bias=False)

        # [D_red, D]
        self.wo = nn.Linear(self.dim_reduced, self.dim, bias=False)

    def forward(self, x):

        x = f(x)

        q = self.wq(x) # [B, N, D]
        k = self.wk(x)
        v = self.wv(x)

        local_heads = self.dim_reduced // self.head_dim

        q = einops.rearrange(q, "b n (h d) -> b h n d", h=local_heads, d=self.head_dim)
        k = einops.rearrange(k, "b n (h d) -> b h n d", h=local_heads, d=self.head_dim)
        v = einops.rearrange(v, "b n (h d) -> b h n d", h=local_heads, d=self.head_dim)

        # do attention
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=(1/math.sqrt(self.head_dim)))

        y = einops.rearrange(y, "b h n d -> b n (h d)", h=local_heads, d=self.head_dim) # [B, N, D_red]

        o = self.wo(y)

        o = g(o)

        return o