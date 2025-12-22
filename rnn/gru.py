import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

from dataclasses import dataclass

from tqdm import tqdm

from rnn.utils import sample

@dataclass
class GRUConfig:

    dim = 64
    layers = 4

    vocab_size = 32000


class GRUBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.dim

        self.gate_network = nn.Linear(2 * self.dim, 2 * self.dim, bias=True)
        self.candidate_network = nn.Linear(2 * self.dim, self.dim, bias=True)

    def forward(self, x, h):
        # x: [B, D], h: [B, D]
        # returns: output, h_new

        xh = torch.cat([x, h], dim=1) # [B, 2D]
        z, r = self.gate_network(xh).chunk(2, dim=-1) # [B, D] each

        z = torch.sigmoid(z)
        r = torch.sigmoid(r)

        # produce candidate h
        candidate = torch.cat([x, r * h], dim=1) # [B, 2D]
        candidate = torch.tanh(self.candidate_network(candidate))

        h_new = z * h + (1 - z) * candidate

        return h_new, h_new


class GRU(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embed = nn.Embedding(self.config.vocab_size, self.config.dim)
        
        self.layers = nn.ModuleList(
            GRUBlock(self.config) for _ in range(self.config.layers)
        )

        self.output = nn.Linear(self.config.dim, self.config.vocab_size, bias=False)

        # Only hidden state starter (no cell state in GRU)
        self.h_starter = nn.Embedding(self.config.layers, self.config.dim)


    def forward(self, input_ids, return_state_list=False):

        B, L = input_ids.shape
        
        x = self.embed(input_ids)  # [B, L, D]

        # init with starter states (hidden only)
        h_list = self.h_starter.weight
        h_list = einops.repeat(h_list, "l d -> b l d", b=B)
        h_list = [h_list[:, i, :] for i in range(self.config.layers)]

        ys = list()
        for i in range(L):
            y, h_list = self.update(x[:, i, :], h_list)  # [B, D], list([B, D])
            y = y.unsqueeze(1)  # [B, 1, D]
            ys.append(y)

        ys = torch.cat(ys, dim=1)
        logits = self.output(ys)

        if return_state_list:
            return logits, h_list

        return logits


    def update(self, x, h_list):
        new_h_list = list()  # collect per layer hidden states

        for i, block in enumerate(self.layers):
            x, h = block(x, h_list[i])
            new_h_list.append(h)

        return x, new_h_list


@torch.inference_mode()
def generate(model: GRU, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.95):

    logits, h_list = model(input_ids, return_state_list=True)

    # decode this token
    # next_token = logits[:, -1, :].argmax(dim=-1) # [B]
    next_token = sample(logits[:, -1, :], temperature, top_k, top_p)

    # add this to list
    decoded_tokens = list()
    decoded_tokens.append(next_token.clone())

    for _ in tqdm(range(max_new_tokens - 1)):
        
        x = model.embed(next_token)  # [B, D]
        
        y, h_list = model.update(x, h_list)

        logits = model.output(y)  # [B, V]

        next_token = sample(logits, temperature, top_k, top_p)
        decoded_tokens.append(next_token.clone())

    decoded_tokens = torch.cat([t.unsqueeze(1) for t in decoded_tokens], dim=1)
    return decoded_tokens


if __name__ == "__main__":
    
    config = GRUConfig()

    model = GRU(config)

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

    decoded_tokens = generate(model, input_ids[:, :10], 12)
    print(decoded_tokens.shape)