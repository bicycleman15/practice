import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

from dataclasses import dataclass

from tqdm import tqdm

from rnn.utils import sample

@dataclass
class LSTMConfig:

    dim = 64
    layers = 4

    vocab_size = 32000


class LSTMBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.dim

        self.gate_network = nn.Linear(2 * self.dim, 4 * self.dim, bias=True) # we keep this true


    def forward(self, x, h, c):
        # x: [B, D], h: [B, D], c: [B, D]
        # returns: h_new, h_new, c_new

        xh = torch.cat([x, h], dim=-1) # [B, 2D]
        i, f, g, o = self.gate_network(xh).chunk(4, dim=-1) # [B, D] each

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_new = f * c + i * g

        h_new = o * torch.tanh(c_new)
        return h_new, h_new, c_new
        

class LSTM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embed = nn.Embedding(self.config.vocab_size, self.config.dim)
        
        self.layers = nn.ModuleList(
            LSTMBlock(self.config) for _ in range(self.config.layers)
        )

        self.output = nn.Linear(self.config.dim, self.config.vocab_size, bias=False)

        # Separate starters for hidden and cell states
        self.h_starter = nn.Embedding(self.config.layers, self.config.dim)
        self.c_starter = nn.Embedding(self.config.layers, self.config.dim)


    def forward(self, input_ids, return_state_list=False):

        B, L = input_ids.shape
        
        x = self.embed(input_ids) # [B, L, D]

        # init with starter states (hidden and cell)
        h_list = self.h_starter.weight
        h_list = einops.repeat(h_list, "l d -> b l d", b=B)
        h_list = [h_list[:, i, :] for i in range(self.config.layers)]

        c_list = self.c_starter.weight
        c_list = einops.repeat(c_list, "l d -> b l d", b=B)
        c_list = [c_list[:, i, :] for i in range(self.config.layers)]

        ys = list()
        for i in range(L):
            y, h_list, c_list = self.update(x[:, i, :], h_list, c_list) # [B, D], list([B, D]), list([B, D])
            y = y.unsqueeze(1) # [B, 1, D]
            ys.append(y)

        ys = torch.cat(ys, dim=1)
        logits = self.output(ys)

        if return_state_list:
            return logits, h_list, c_list

        return logits


    def update(self, x, h_list, c_list):
        new_h_list = list()  # collect per layer hidden states
        new_c_list = list()  # collect per layer cell states

        for i, block in enumerate(self.layers):
            x, h, c = block(x, h_list[i], c_list[i])
            new_h_list.append(h)
            new_c_list.append(c)

        return x, new_h_list, new_c_list


@torch.inference_mode()
def generate(model: LSTM, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.95):

    logits, h_list, c_list = model(input_ids, return_state_list=True)

    # decode this token
    # next_token = logits[:, -1, :].argmax(dim=-1) # [B]
    next_token = sample(logits[:, -1, :], temperature, top_k, top_p)

    # add this to list
    decoded_tokens = list()
    decoded_tokens.append(next_token.clone())

    for _ in tqdm(range(max_new_tokens - 1)):
        
        x = model.embed(next_token) # [B, D]
        
        y, h_list, c_list = model.update(x, h_list, c_list)

        logits = model.output(y) # [B, V]

        next_token = sample(logits, temperature, top_k, top_p)
        decoded_tokens.append(next_token.clone())

    decoded_tokens = torch.cat([t.unsqueeze(1) for t in decoded_tokens], dim=1)
    return decoded_tokens


if __name__ == "__main__":
    
    config = LSTMConfig()

    model = LSTM(config)

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