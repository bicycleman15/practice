import torch
import einops

B = 4
L = 128

D = 1024
H = 16
d = 64

assert H*d == D

# query vector in HBM
q = torch.randn((B, 1, D)) # [B, 1, D]
q = einops.reshape(q, "b 1 (h d) -> b h 1 d", h=H, d=d)

# key value cache in HBM
k = torch.randn((B, H, L, d))
v = torch.randn((B, H, L, d))

### kernel starts below
### everything below happens in SRAM

# move q to SRAM
# create buffers for denominator and output vector
y = torch.zeros((B, H, 1, d)) # stays in SRAM
den_sum = torch.zeros((B, H, 1, 1))
den_max = torch.ones((B, H, 1, 1)) * (-1e9)

block_size = 16

for i in range(0, L, block_size):

    start = i
    end = i + block_size

    # load key-values from HBM to SRAM
    key = k[:, :, start:end, :]
    value = v[:, :, start:end, :]

    # do attention
    s = q @ key.T # [B, H, 1, block_size]

    cur_max = torch.max(s, dim=-1, keepdim=True) # [B, H, 1, 1]
    cur_max = torch.maximum(cur_max, den_max) # [B, H, 1, 1]

    # correct den_sum
    den_sum = den_sum * torch.exp(den_max - cur_max) + torch.exp(s - cur_max).sum(dim=-1, keepdim=True) # [B, H, 1, 1]

    # [B, H, 1, block_size] @ [B, H, block_size, d] = [B, H, 1, d]
    y = y * torch.exp(den_max - cur_max) + ((torch.exp(s - cur_max) / den_sum) @ value) 

    den_max = cur_max

return y
