import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

from dataclasses import dataclass

from tqdm import tqdm

from rnn.utils import sample

@dataclass
class RNNConfig:

    dim = 64
    layers = 4

    vocab_size = 32000


class RNNBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.update_network = nn.Linear(2 * config.dim, 2 * config.dim, bias=False)

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

        self.output = nn.Linear(self.config.dim, self.config.vocab_size, bias=False)

        self.starter = nn.Embedding(self.config.layers, self.config.dim)


    def forward(self, input_ids, return_state_list=False):

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

        if return_state_list:
            return logits, s_list

        return logits


    def update(self, x, s_list):
        new_s_list = list() # collect per layer states

        for i, block in enumerate(self.layers):
            x, s = block(x, s_list[i])
            new_s_list.append(s)

        return x, new_s_list


@torch.inference_mode()
def generate(model: RNN, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.95):

    logits, s_list = model(input_ids, return_state_list=True)

    # decode this token
    # next_token = logits[:, -1, :].argmax(dim=-1) # [B]
    next_token = sample(logits[:, -1, :], temperature, top_k, top_p)

    finished = torch.full((logits.shape[0]), False, dtype=torch.bool)

    # add this to list
    decoded_tokens = list()
    decoded_tokens.append(next_token.clone())

    for _ in tqdm(range(max_new_tokens - 1)):
        
        x = model.embed(next_token) # [B, D]
        
        y, s_list_new = model.update(x, s_list)

        logits = model.output(y) # [B, V]

        next_token = sample(logits, temperature, top_k, top_p)

        # just finished
        just_finished = (next_token == 0) # 0 is EOS token id
        finished = finished | just_finished

        # change next token accordingly
        # 
        # EOS token feeded again might result in different next tokens
        # for sequence alreay finished, so make sure they always stay EOS
        next_token = torch.where(finished, 0, next_token)

        # change states accordingly
        alive = 1 - finished
        for i in range(len(s_list)):
            s_list[i] = alive * s_list_new[i] + (1 - alive) * s_list[i]

        decoded_tokens.append(next_token.clone())

        if torch.all(finished):
            break

    decoded_tokens = torch.cat([t.unsqueeze(1) for t in decoded_tokens], dim=1)
    return decoded_tokens


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
    bar = tqdm(range(2))
    for _ in bar:
        optimizer.zero_grad()

        logits = model(input_ids)

        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), input_ids.view(-1))

        loss.backward()
        optimizer.step()
        bar.set_postfix_str(f"loss={loss.item():02f}")

    decoded_tokens = generate(model, input_ids[:, :10], 12)
    print(decoded_tokens.shape)