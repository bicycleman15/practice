"""
My attempt at writing torch DDP from scratch
"""

import torch
import torch.distributed as dist

class DDP:
    def __init__(self, model: torch.nn.Module, rank: int):
        super().__init__()

        self.model = model

        self.bucket_size = 10
        self.buckets = None

        # we must sync parameters across all ranks
        self.sync_weights()

        self.divide_into_buckets()
        self.register_hooks()

    def sync_weights(self):
        for name, param in self.model.named_parameters():
            dist.broadcast(param.data, src=0)

    def forward(self, *args, **kwargs):
        # this will run in parallel in all ranks without any comms
        return self.model.forward(*args, **kwargs)

    def divide_into_buckets(self):
        # i think these should be done during autograd
        # since we might accidently decide a buckets such that not
        # even a single all_reduce is overlapped at all :)

        # for now we manually do it

        buckets = list()
        cur_bucket = list()

        for name, param in self.model.named_parameters():

            # push this param to cur_bucket
            cur_bucket.append(param)
            param2bucketidx[id(param)] = len(buckets) - 1

            if len(cur_bucket) == self.bucket_size:
                # push this cur_bucket to all buckets
                buckets.append(cur_bucket)
                self.last_bucket_size = len(cur_bucket)
                cur_bucket = list()

        if len(cur_bucket) > 0:
            # push the leftover
            buckets.append(cur_bucket)
            self.last_bucket_size = len(cur_bucket)
            cur_bucket = list()

        self.buckets = buckets


    def register_hooks(self):

        def this_is_the_hook(param: torch.Tensor):

            # add +1 to param in this bucket that has just done backward right now
            idx = self.param2bucketidx[id(param)]
            self.bucket_counter[idx] += 1

            last_bucket_done = (idx == len(self.buckets) - 1) and (self.bucket_counter[idx] == self.last_bucket_size)

            # if all params in this current bucket are done
            # reduce params in this bucket
            if self.bucket_counter[idx] == self.bucket_size or last_bucket_done:

                for param in self.buckets[idx]:
                    work = dist.all_reduce(param.grad, op=AVG, async=True) # only returns work obj only when async=True
                    # push this to all works
                    self.works.append(work)

                # reset the flag that checks for this
                self.bucket_counter[idx] = 0

        
        for name, param in self.model.named_parameters():
            # NOTE: this is not right syntax, but the idea is correct :))
            # register this hook for this param
            # so that above function runs after backward has been called on this param
            param.hook = this_is_the_hook

    def synhronize():

        for work in self.works:
            work.wait()

        # reset works
        self.works = list()