import math
import torch
from tqdm import trange

@torch.no_grad()
def flash_attention(q, k, v, scale):
    # q: [B, H, N, D]

    block_size = 32
    N = q.shape[2]

    out = torch.zeros_like(q) # q: [B, H, N, D]
    B, H, N, D = q.shape

    all_den_sum = torch.zeros_like(q)[..., -1].unsqueeze(-1) # [B, H, N, 1]
    max_logit = torch.zeros_like(q)[..., -1].unsqueeze(-1) # [B, H, N, 1]

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
        all_den_sum[:, :, i:i+block_size, :] = den_sum
        max_logit[:, :, i:i+block_size, :] = max_temp
    
    return out, all_den_sum, max_logit

@torch.no_grad()
def flash_attention_backward_rowwise(q, k, v, scale, do, o, sum_denom, max_logit):
    
    # q: [B, H, N, D]
    N = q.shape[2]

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(q)
    dv = torch.zeros_like(q)

    # compute helper scalar for ds
    helper_scalar = torch.sum(do * o, dim=-1, keepdim=True) # [:, :, N, 1]

    for i in range(N):

        block_size = 16

        q_tile = q[:, :, i:i+1] # [:, :, 1, D]

        sum_row = sum_denom[:, :, i:i+1] # [:, :, 1, 1]
        max_row = max_logit[:, :, i:i+1]

        cur_do = do[:, :, i:i+1] # [:, :, 1, D]

        cur_helper_scalar = helper_scalar[:, :, i:i+1] # [:, :, 1, 1]

        for j in range(0, N, block_size):

            k_tile = k[:, :, j:j+block_size]
            v_tile = v[:, :, j:j+block_size]

            s = q_tile @ k_tile.transpose(-1, -2) / scale # [:, :, 1, T]
            p = torch.exp(s - max_row) / sum_row

            dv[:, :, j:j+block_size] += p.transpose(-1, -2) @ cur_do # [:, :, T, D]

            dp = cur_do @ v_tile.transpose(-1, -2) # [:, :, 1, T]

            ds = dp * p - cur_helper_scalar * p # [:, :, 1, T]

            dk[:, :, j:j+block_size] += ds.transpose(-1, -2) @ q_tile / scale # [:, :, T, D]

            dq[:, :, i:i+1] += ds @ k_tile / scale # [:, :, 1, D]

    return dq, dk, dv


if __name__ == "__main__":

    B, H, N, D = 8, 2, 512, 64

    for _ in trange(1):

        # Create tensors with gradients enabled for autograd comparison
        q = torch.randn((B, H, N, D), requires_grad=True)
        k = torch.randn((B, H, N, D), requires_grad=True)
        v = torch.randn((B, H, N, D), requires_grad=True)
        do = torch.randn((B, H, N, D))

        scale = math.sqrt(q.shape[-1])

        # === Forward pass test ===
        out, den_sum, max_logit = flash_attention(q, k, v, scale)
        
        # Reference implementation using PyTorch
        out_torch = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=(1/scale)
        )
        
        # Check forward pass correctness
        fwd_error = (out - out_torch).abs().max().item()
        print(f"Forward pass max error: {fwd_error:.2e}")
        assert torch.allclose(out, out_torch, atol=1e-5, rtol=1e-5), \
            f"Forward pass failed! Max error: {fwd_error}"
        print("Forward pass test PASSED!")

        # === Backward pass test ===
        # Compute reference gradients using autograd
        out_torch.backward(do)
        dq_ref = q.grad.clone()
        dk_ref = k.grad.clone()
        dv_ref = v.grad.clone()

        # Compute gradients using custom backward (use detached tensors)
        dq, dk, dv = flash_attention_backward_rowwise(
            q.detach(), k.detach(), v.detach(), scale, do, out, den_sum, max_logit
        )

        # Check backward pass correctness
        dq_error = (dq - dq_ref).abs().max().item()
        dk_error = (dk - dk_ref).abs().max().item()
        dv_error = (dv - dv_ref).abs().max().item()

        print(f"dQ max error: {dq_error:.2e}")
        print(f"dK max error: {dk_error:.2e}")
        print(f"dV max error: {dv_error:.2e}")

        assert torch.allclose(dq, dq_ref, atol=1e-5, rtol=1e-5), \
            f"dQ backward failed! Max error: {dq_error}"
        assert torch.allclose(dk, dk_ref, atol=1e-5, rtol=1e-5), \
            f"dK backward failed! Max error: {dk_error}"
        assert torch.allclose(dv, dv_ref, atol=1e-5, rtol=1e-5), \
            f"dV backward failed! Max error: {dv_error}"
        print("Backward pass test PASSED!")

    print("\nAll tests passed!")