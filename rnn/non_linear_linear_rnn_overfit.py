"""
Overfit a single batch with the NonLinearLinearRNNLM across all four boundary
non-linearity variants, to sanity-check that each configuration can actually
drive the loss toward zero.

Run with:
    source ~/.zshrc && conda activate t3
    python -m rnn.non_linear_linear_rnn_overfit
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

from rnn.non_linear_linear_rnn import NonLinearLinearRNNConfig, NonLinearLinearRNNLM


@dataclass
class OverfitConfig:
    batch_size: int = 2
    seqlen: int = 128
    steps: int = 200
    lr: float = 3e-3
    seed: int = 0
    log_every: int = 10          # update tqdm postfix every N steps (keeps non-TTY output tidy)

    # model
    dim: int = 64
    num_heads: int = 2
    head_k_dim: int = 16
    head_v_dim: int = 16
    layers: int = 2
    vocab_size: int = 256
    chunk_size: int = 16
    use_short_conv: bool = True


def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def overfit_one_variant(boundary_nonlin: str, cfg: OverfitConfig, device: str):
    torch.manual_seed(cfg.seed)

    model_config = NonLinearLinearRNNConfig(
        dim=cfg.dim,
        num_heads=cfg.num_heads,
        head_k_dim=cfg.head_k_dim,
        head_v_dim=cfg.head_v_dim,
        layers=cfg.layers,
        vocab_size=cfg.vocab_size,
        chunk_size=cfg.chunk_size,
        boundary_nonlin=boundary_nonlin,
        use_short_conv=cfg.use_short_conv,
    )
    model = NonLinearLinearRNNLM(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    input_ids = torch.randint(
        0, cfg.vocab_size, (cfg.batch_size, cfg.seqlen), device=device,
    )

    initial_loss = math.log(cfg.vocab_size)
    bar = tqdm(
        range(cfg.steps),
        desc=f"overfit [{boundary_nonlin:<9s}]",
        mininterval=0.5,
        miniters=cfg.log_every,
    )
    final_loss = float("nan")
    min_loss = float("inf")
    for step in bar:
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            input_ids.view(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        final_loss = loss.item()
        min_loss = min(min_loss, final_loss)
        if (step + 1) % cfg.log_every == 0 or step == cfg.steps - 1:
            bar.set_postfix_str(
                f"loss={final_loss:.4f} min={min_loss:.4f} (init~{initial_loss:.2f})"
            )

    with torch.no_grad():
        logits = model(input_ids)
        preds = logits.argmax(dim=-1)
        acc = (preds == input_ids).float().mean().item()

    return {
        "variant": boundary_nonlin,
        "final_loss": final_loss,
        "min_loss": min_loss,
        "token_acc": acc,
    }


def main():
    cfg = OverfitConfig()
    device = _pick_device()
    print(f"device={device}  cfg={cfg}\n")

    results = []
    for kind in ["identity", "rmsnorm", "tanh_res", "gru"]:
        results.append(overfit_one_variant(kind, cfg, device))

    print("\n=== overfit summary ===")
    print(f"  initial CE loss (uniform over {cfg.vocab_size} tokens) ~ {math.log(cfg.vocab_size):.3f}")
    header = f"  {'variant':<10s} {'final_loss':>12s} {'min_loss':>12s} {'token_acc':>12s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        print(
            f"  {r['variant']:<10s} "
            f"{r['final_loss']:>12.4f} "
            f"{r['min_loss']:>12.4f} "
            f"{r['token_acc']:>12.2%}"
        )


if __name__ == "__main__":
    main()
