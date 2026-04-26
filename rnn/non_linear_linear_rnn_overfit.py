"""
Overfit test for M²RNN on S_n state tracking.

Train on a SINGLE fixed batch and check that the model can memorize it
(loss -> 0, accuracy -> 100%). This isolates "the model can fit" from
"the model can generalize". If overfit fails, there's an architectural
bug; if it succeeds, the issue with the full benchmark is training scale
or generalization, not the impl.

Default: S_3 (paper's actual benchmark group), B=4, L=128, 2000 steps.
Comparison: also runs the linear_chunks/identity baseline as a sanity-
check that the training loop itself works on the same fixed batch.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

from rnn.non_linear_linear_rnn import (
    NonLinearLinearRNNConfig,
    NonLinearLinearRNNLM,
)
from rnn.non_linear_linear_rnn_s3 import (
    SnConfig,
    _build_sn_cayley,
    make_batch,
    _pick_device,
)
from math import factorial


@dataclass
class OverfitConfig:
    group_n: int = 3
    L: int = 128
    batch_size: int = 4
    steps: int = 2000
    lr: float = 1e-2
    seed: int = 0
    log_every: int = 100

    dim: int = 128
    num_heads: int = 2
    head_k_dim: int = 32
    head_v_dim: int = 32
    layers: int = 1

    # If True, also run linear_chunks/identity baseline as a sanity check
    # that the training loop / data pipeline works on the same fixed batch.
    # Skip this once you've seen it converge once.
    run_baseline: bool = False
    # Early-stop a run as soon as accuracy stays >= this for `early_stop_k`
    # consecutive steps. Saves time when the model converges quickly.
    early_stop_acc: float = 0.999
    early_stop_k: int = 20


def run_overfit(architecture: str, boundary_nonlin: str, oc: OverfitConfig,
                cayley_t, identity_idx, vocab_size, device,
                input_ids, target_ids):
    torch.manual_seed(oc.seed)
    cfg = NonLinearLinearRNNConfig(
        dim=oc.dim,
        num_heads=oc.num_heads,
        head_k_dim=oc.head_k_dim,
        head_v_dim=oc.head_v_dim,
        layers=oc.layers,
        vocab_size=vocab_size,
        chunk_size=1,
        boundary_nonlin=boundary_nonlin,
        use_short_conv=True,
        decay_init="mild",
        allow_neg_eigval=True,
        architecture=architecture,
    )
    model = NonLinearLinearRNNLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optim = torch.optim.AdamW(model.parameters(), lr=oc.lr)

    tag = architecture if architecture == "m2rnn" else f"{architecture}/{boundary_nonlin}"
    print(f"\n--- overfit run: {tag}  ({n_params/1e6:.2f}M params) ---")

    bar = tqdm(range(oc.steps), desc=tag, mininterval=0.5)
    last_acc = 0.0
    last_loss = float("inf")
    streak = 0
    for step in bar:
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), target_ids.reshape(-1))
        optim.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        with torch.no_grad():
            acc = (logits.argmax(-1) == target_ids).float().mean().item()
        last_acc = acc
        last_loss = loss.item()
        bar.set_postfix_str(f"loss={last_loss:.4f} acc={acc:.2%} gn={gn:.3f}")

        if acc >= oc.early_stop_acc:
            streak += 1
            if streak >= oc.early_stop_k:
                bar.close()
                print(f"  early-stop at step {step+1}: acc={acc:.2%} for {streak} steps")
                break
        else:
            streak = 0

    return last_loss, last_acc


def main():
    oc = OverfitConfig()
    device = _pick_device()
    perms, cayley = _build_sn_cayley(oc.group_n)
    vocab_size = factorial(oc.group_n)
    identity_idx = perms.index(tuple(range(oc.group_n)))
    cayley_t = torch.tensor(cayley, dtype=torch.long)

    print(f"device={device}")
    print(f"S_{oc.group_n}  vocab={vocab_size}  L={oc.L}  B={oc.batch_size}  steps={oc.steps}")
    print(f"random-guess accuracy: {1/vocab_size:.2%}")

    # Build ONE fixed batch and reuse it every step.
    torch.manual_seed(oc.seed)
    input_ids, target_ids = make_batch(
        oc.batch_size, oc.L, vocab_size, cayley_t, identity_idx, device,
    )

    runs = []
    if oc.run_baseline:
        # Sanity-check: the simplest delta-rule + identity baseline. Should
        # easily memorize a small fixed batch. If this fails, the training
        # loop itself is broken.
        runs.append(("linear_chunks", "identity"))
    # M²RNN under test.
    runs.append(("m2rnn", "m2rnn"))
    results = {}
    for arch, kind in runs:
        loss, acc = run_overfit(arch, kind, oc, cayley_t, identity_idx,
                                 vocab_size, device, input_ids, target_ids)
        tag = arch if arch == "m2rnn" else f"{arch}/{kind}"
        results[tag] = (loss, acc)

    print("\n=== overfit summary ===")
    for tag, (loss, acc) in results.items():
        verdict = "PASS" if acc > 0.99 else "FAIL"
        print(f"  {tag:<25s}  final loss={loss:.4f}  acc={acc:.2%}  [{verdict}]")


if __name__ == "__main__":
    main()
