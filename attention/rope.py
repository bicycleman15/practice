import torch

def create_rope_cache(seqlen, head_dim, base = 10_000):
    
    pos = torch.arange(seqlen) # [N]

    multiplier_across_dims = 1 / torch.arange(head_dim // 2) # [D/2]

    angles = pos[:, None] * multiplier_across_dims[None, :] # [N, D/2]

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def apply_rope(x, cos, sin):
    
    B, H, N, D = x.shape

    x_even = x[..., ::2] # [B, H, N, D/2]
    x_odd = x[..., 1::2]

    cos = cos[None, None, ...] # [1, 1, N, D/2]
    sin = sin[None, None, ...]

    x_first = x_even * cos - x_odd * sin
    x_second = x_even * cos + x_odd * sin

    # stitch back together
    y = torch.empty_like(x)
    y[..., ::2] = x_first
    y[..., 1::2] = x_second

    return y.contigous()
    