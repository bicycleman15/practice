import torch

base = 1000

N = 1024
D = 256

dims = torch.arange(0, D, 2, dtype=torch.float)[None, :]
pos = torch.arange(N)[:, None]

pos_emb = pos / (base ** (dims / D))

pos_emb[:, 0::2] = torch.sin(pos_emb[:, 0::2])
pos_emb[:, 1::2] = torch.cos(pos_emb[:, 1::2])

print(pos_emb.shape)