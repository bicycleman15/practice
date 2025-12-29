from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist

@dataclass
class ConfigTP:
    num_processes: int = 2

    dim: int = 256

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

