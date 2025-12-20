import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

from dataclasses import dataclass

from tqdm import tqdm

@dataclass
class RNNConfig:

    dim = 64
    layers = 4

    vocab_size = 1024


class RNNBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.update_network = nn.Linear(2 * config.dim, 2 * config.dim)

    def forward(self, x, s):
        xs = torch.cat([x, s], dim=1) # [B, 2D]
        xs = self.update_network(xs) # [B, 2D]

        x, s = torch.chunk(xs, 2, dim=1)

        return x, s


class RNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embed = nn.Embedding(self.config.vocab_size, self.config.dim)
        
        self.layers = nn.ModuleList(
            RNNBlock(self.config) for _ in range(self.config.layers)
        )

        self.output = nn.Linear(self.config.dim, self.config.vocab_size)

        self.starter = nn.Embedding(self.config.layers, self.config.dim)


    def forward(self, input_ids):

        B, L = input_ids.shape
        
        x = self.embed(input_ids) # [B, L, D]

        # init with starter states
        s_list = self.starter.weight
        s_list = einops.repeat(s_list, "l d -> b l d", b=B)

        s_list = [s_list[:, i, :] for i in range(self.config.layers)]

        ys = list()
        for i in range(L):
            y, s_list = self.update(x[:, i, :], s_list) # [B, D], list([B, D])
            y = y.unsqueeze(1) # [B, 1, D]
            ys.append(y)

        ys = torch.cat(ys, dim=1)
        logits = self.output(ys)

        return logits


    def update(self, x, s_list):
        new_s_list = list() # collect per layer states

        for i, block in enumerate(self.layers):
            x, s = block(x, s_list[i])
            new_s_list.append(s)

        return x, new_s_list


if __name__ == "__main__":
    
    config = RNNConfig()

    model = RNN(config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch_size = 2
    seqlen = 256

    input_ids = torch.randint(0, config.vocab_size, size=(batch_size, seqlen))

    print(input_ids.shape)

    logits = model(input_ids)

    print(logits.shape)

    # calc loss
    bar = tqdm(range(1000))
    for _ in bar:
        optimizer.zero_grad()

        logits = model(input_ids)

        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), input_ids.view(-1))

        loss.backward()
        optimizer.step()
        bar.set_postfix_str(f"loss={loss.item():02f}")