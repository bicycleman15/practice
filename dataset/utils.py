import os
import torch
import numpy as np
from torch.utils.data import IterableDataset

class tiny_stories_dataset(IterableDataset):

    def __init__(self, block_size, batch_size, split="train", path="dataset/tiny_stories"):
        super().__init__()

        # load the npy
        self.tokens = np.load(os.path.join(path, f"{split}.npy"))
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        self.block_size = block_size + 1 # since we need an extra token for next-token prediction
        self.batch_size = batch_size

        print(f"Loading the {split} split: containing {self.tokens.shape[0]} tokens ...")

        # load the tokens and shuffle sequences
        cut_off = (self.tokens.shape[0] // self.block_size) * self.block_size
        self.tokens = self.tokens[:cut_off]

        self.tokens = self.tokens.reshape(-1, self.block_size)

        # permute sequences
        # fix seed of randperm to some constant
        g = torch.Generator().manual_seed(42)
        perm = torch.randperm(self.tokens.shape[0], generator=g)
        self.tokens = self.tokens[perm, :]
        self.tokens = self.tokens.reshape(-1)

        self.idx = 0
        self.stride = self.block_size * self.batch_size


    def __len__(self):
        return self.tokens.shape[0]

    
    def __iter__(self):

        while True:
            if self.idx + self.stride >= len(self):
                self.idx = 0 # reset self.idx

            # take a slice :)
            batch = self.tokens[self.idx : self.idx + self.stride].clone()
            batch = batch.reshape(-1, self.block_size)
            self.idx += self.stride

            input_ids = batch[:, :-1].clone()
            targets = batch[:, 1:].clone()

            yield input_ids, targets


if __name__ == "__main__":

    data = tiny_stories_dataset(
        block_size=256,
        batch_size=2,
        split="train"
    )
    iterdata = iter(data)
    bb = next(iterdata)

    print(bb[0].shape)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    print(tokenizer.decode(bb[0][0], ))
    breakpoint()

    print("Done...")