import math
import torch
from tqdm import trange


def flash_attention(q, k, v, scale):
    # q: [B, H, N, D]

    block_size = 32
    N = q.shape[2]

    out = torch.zeros_like(q) # q: [B, H, N, D]
    B, H, N, D = q.shape

    for i in range(0, N, block_size):

        q_tile = q[:, :, i:i+block_size, :] # [T, D]
        T = q_tile.shape[2]

        out_temp = torch.zeros_like(q_tile) # [T, D]
        max_temp = torch.ones((B, H, T, 1), dtype=q_tile.dtype, device=q_tile.device) * 1e-12 # [T, 1]
        den_sum = torch.zeros((B, H, T, 1), dtype=q_tile.dtype, device=q_tile.device)

        for j in range(0, N, block_size):

            k_tile = k[:, :, j:j+block_size, :] # [T, D]
            v_tile = v[:, :, j:j+block_size, :] # [T, D]

            # take dot product
            dot = q_tile @ k_tile.transpose(-1, -2) / scale # [T, T]

            cur_max = torch.max(dot, dim=-1, keepdim=True)[0] # [T, 1]
            cur_max = torch.maximum(cur_max, max_temp) # [T, 1]

            scores = torch.exp(dot - cur_max) # [T, T]

            new_v = scores @ v_tile # [T, T] @ [T, D] = [T, D]

            out_temp = out_temp * torch.exp(max_temp - cur_max) + new_v # [T, D]
            den_sum = den_sum * torch.exp(max_temp - cur_max) + scores.sum(dim=-1, keepdim=True) # [T, 1]
            
            max_temp = cur_max

        out[:, :, i:i+block_size, :] = out_temp / den_sum
    
    return out


if __name__ == "__main__":

    B, H, N, D = 8, 2, 512, 64

    for _ in trange(100):

        q = torch.randn((B, H, N, D))
        k = torch.randn((B, H, N, D))
        v = torch.randn((B, H, N, D))

        scale = math.sqrt(q.shape[-1])

        out = flash_attention(q, k, v, scale)
        # print(out.shape)
        # print(out[0, 0, :5, :5])

        out_torch = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=(1/scale))
        # print(out_torch.shape)
        # print(out_torch[0, 0, :5, :5])

        assert torch.allclose(out, out_torch, atol=1e-5, rtol=1e-5)

    print("Tests passed!")