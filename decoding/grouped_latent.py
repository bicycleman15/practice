import torch
import einops

B = 4
L = 128

D = 1024
H = 16
d = 64
c = 256 # usually chosen as 4 * d
latent_heads = 2

# now there are these weights
W_q
W_c_to_k # instead of W_k
W_c_to_v # instead of W_k
W_o
assert H*d == D

# Assume single GPU setup only!

# query vector in HBM
q = torch.randn((B, 1, D)) # [B, 1, D]
q = einops.reshape(q, "b 1 (h d) -> b h 1 d", h=H, d=d)

# NOTE
q_c = q @ w_c_to_k.T # # [B, latent_heads, c]

# NOTE
# latent cache in HBM
latent_cache = torch.randn((B, 1, c))

### kernel starts below
### everything below happens in SRAM

# move q to SRAM
# create buffers for denominator and output vector
# NOTE !!!
y = torch.zeros((B, latent_heads, c)) # stays in SRAM
den_sum = torch.zeros((B, latent_heads, 1, 1))
den_max = torch.ones((B, latent_heads, 1, 1)) * (-1e9)

block_size = 16

for i in range(0, L, block_size):

    start = i
    end = i + block_size

    # load cache from HBM to SRAM
    # NOTE: we load less stuff from HBM
    cache = latent_cache[:, :, start:end, :]

    # now here, we could map to keys-values
    # and then duplicate keys-values to match heads of query

    # (q @ (c @ w_c_to_k).T) = (q @ w_c_to_k.T) @ c.T 
    # i.e. "do attention in latent space"

    # do attention
    # NOTE: the dot product is happening with 256 sized vectors, rather than 64
    # there's some trade-off with block_size
    s = q @ cache.T # [B, H, 1, block_size]

    cur_max = torch.max(s, dim=-1, keepdim=True) # [B, latent_heads, 1, 1]
    cur_max = torch.maximum(cur_max, den_max) # [B, latent_heads, 1, 1]

    # correct den_sum
    den_sum = den_sum * torch.exp(den_max - cur_max) + torch.exp(s - cur_max).sum(dim=-1, keepdim=True) # [B, H, 1, 1]

    # [B, H, 1, block_size] @ [B, latent_heads, block_size, d] = [B, latent_heads, 1, d]
    y = y * torch.exp(den_max - cur_max) + ((torch.exp(s - cur_max) / den_sum) @ value) 

    den_max = cur_max

# NOTE
return y # [B, latent_heads, c]


# NOTE: now later, do
o = y @ (w_to_v @ w_o) # [B, H, D]
