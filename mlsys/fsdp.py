# only supports linear layers (without bias) for now
# we had to write the backward for linear ourselves, but thats okay for a toy implementation
# the real implementation would actually use the pytorch autograd after all-gathering the weights
# then arranging back to the original module, and taking the pytorch autograd backward

import torch
import torch.distributed as dist

class FSDPForwardBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight_shard, world_size, rank):
        # All-gather weights from all ranks
        weight_shape = weight_shard.shape
        full_weight = torch.empty(
            weight_shape[0] * world_size, weight_shape[1],
            dtype=weight_shard.dtype, device=weight_shard.device
        )
        
        # Gather all weight shards
        weight_list = list(torch.chunk(full_weight, world_size, dim=0))
        dist.all_gather(weight_list, weight_shard)
        full_weight = torch.cat(weight_list, dim=0)
        
        # Forward pass: y = W @ x
        y = torch.mm(full_weight, x)
        
        # Save for backward
        ctx.save_for_backward(x, weight_shard)
        ctx.world_size = world_size
        ctx.rank = rank
        ctx.weight_shape = weight_shape
        
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight_shard = ctx.saved_tensors
        world_size = ctx.world_size
        rank = ctx.rank
        weight_shape = ctx.weight_shape
        
        # All-gather weights again for backward pass
        full_weight = torch.empty(weight_shape[0] * world_size, weight_shape[1], 
                                   dtype=weight_shard.dtype, device=weight_shard.device)
        weight_list = list(torch.chunk(full_weight, world_size, dim=0))
        dist.all_gather(weight_list, weight_shard)
        full_weight = torch.cat(weight_list, dim=0)
        
        # Compute gradients
        # grad_x = W^T @ grad_output
        grad_x = torch.mm(full_weight.t(), grad_output)
        
        # grad_W = grad_output @ x^T
        grad_weight_full = torch.mm(grad_output, x.t())
        
        # Reduce-scatter: each rank gets its shard of the gradient
        grad_weight_shard = torch.zeros_like(weight_shard)
        grad_weight_list = list(torch.chunk(grad_weight_full, world_size, dim=0))
        dist.reduce_scatter(grad_weight_shard, grad_weight_list)
        
        return grad_x, grad_weight_shard, None, None


class FSDPLayer(torch.nn.Module):

    def __init__(self, module, rank, world_size):
        super().__init__()

        self.rank = rank
        self.world_size = world_size

        with torch.no_grad():
            # shard the weights for the module
            self.weight = torch.nn.Parameter(
                torch.chunk(module.weight, self.world_size, dim=0)[self.rank]
            )


    def forward(self, x):
        return FSDPForwardBackward.apply(x, self.weight, self.world_size, self.rank)
