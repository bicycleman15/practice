import torch
import torch.nn as nn


def scan_forward(x, coeff):
    # x: [B, L, D]
    # coeff: [B, L, D]

    B, L, D = x.shape

    ys = torch.zeros_like(x)
    prev = torch.zeros_like(x[:, 0]) # [B, D]

    for i in range(L):
        prev = prev * coeff[:, i] + x[:, i] # [B, D]
        ys[:, i] = prev

    return ys

def scan_forward_parallel(x, coeff):
    # x: [B, L, D]
    # coeff: [B, L, D]

    B, L, D = x.shape
    prev = torch.zeros_like(x[:, 0]) # [B, D]

    coeff_prod = torch.cumprod(coeff, dim=1) # [B, L, D]
    cum_sum = (x / coeff_prod).cumsum(dim=1) # [B, L, D]

    ys = coeff_prod * (prev.unsqueeze(1) + cum_sum) # [B, L, D]

    return ys


if __name__ == "__main__":
    import time

    B, D = 8, 64
    seq_lens = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    num_warmup = 3
    num_runs = 10

    print(f"{'SeqLen':<10} {'Sequential (ms)':<18} {'Parallel (ms)':<18} {'Speedup':<10} {'Match':<10}")
    print("-" * 70)

    for L in seq_lens:
        x = torch.randn((B, L, D))
        coeff = torch.randn((B, L, D))

        # Warmup
        for _ in range(num_warmup):
            _ = scan_forward(x, coeff)
            _ = scan_forward_parallel(x, coeff)

        # Benchmark sequential
        start = time.perf_counter()
        for _ in range(num_runs):
            y = scan_forward(x, coeff)
        seq_time = (time.perf_counter() - start) / num_runs * 1000

        # Benchmark parallel
        start = time.perf_counter()
        for _ in range(num_runs):
            y_parallel = scan_forward_parallel(x, coeff)
        par_time = (time.perf_counter() - start) / num_runs * 1000

        # Check correctness
        match = "✅" if torch.allclose(y, y_parallel, atol=1e-4, rtol=1e-3) else "❌"
        max_diff = (y - y_parallel).abs().max().item()
        speedup = seq_time / par_time if par_time > 0 else float('inf')

        print(f"{L:<10} {seq_time:<18.3f} {par_time:<18.3f} {speedup:<10.2f}x {match:<10} max_diff={max_diff:.2e}")