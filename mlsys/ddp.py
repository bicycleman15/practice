"""
My attempt at writing torch DDP from scratch
"""

import torch
import torch.distributed as dist

class DDP:
    def __init__(self, model: torch.nn.Module, rank: int):
        super().__init__()

        self.model = model

        # we must sync parameters across all ranks
        self.sync_weights()

    def sync_weights(self):
        for name, param in self.model.named_parameters():
            dist.broadcast(param.data, src=0)

    def forward(self, *args, **kwargs):
        # this will run in parallel in all ranks without any comms
        return self.model.forward(*args, **kwargs)

    def backward(self, loss):
        # run in parallel
        loss.backward()

        # now, we have diff gradients across ranks
        # we need to all_reduce across all params
        with torch.no_grad(): # just to be safe
            for name, param in self.model.named_parameters():
                # create an empty tensor
                reduced_tensor_grad = torch.empty_like(param.grad)
                dist.all_reduce(param.grad, reduced_tensor_grad, op=dist.ReduceOp.MEAN)

                # replace this param in model
                param.grad = reduced_tensor_grad

        # now we have everything done
