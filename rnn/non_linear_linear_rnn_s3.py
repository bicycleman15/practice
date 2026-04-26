"""
S_n state-tracking length-generalization benchmark (default n=5 -> S_5).

Task:
    input_ids[t]  = random element g_t in S_n            (vocab = n!)
    target_ids[t] = cumulative product g_1 . g_2 . ... . g_t  (vocab = n!)

Why S_n, and why does n matter:
    - S_3 is the smallest non-abelian group; S_5 is the smallest non-solvable
      group. Word problems in non-solvable groups are NC^1-complete and cannot
      be parallelised by a bounded-depth circuit (Barrington 1989).
    - Pure linear RNNs cannot solve any non-trivial group word problem with
      eigenvalues constrained to [0, 1] (Grazzi et al. 2024,
      "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues",
      arXiv:2411.12537). Hence allow_neg_eigval=True is mandatory.
    - One Householder per token (n_h=1) reaches at most determinant -1
      reflections, which is sufficient for parity but NOT for 3-cycles in S_3.
      DeltaProduct (Siems et al. 2025, arXiv:2502.10297) gives the bound:
          S_3, S_4, A_5  -> n_h >= 2
          S_5             -> n_h >= 4
      We have n_h=1 here, so for S_5 the boundary nonlinearity has to do real
      work composing per-chunk group multiplications.

We train on sequences of length L_train and evaluate on lengths
{L_train, 2*L_train, 4*L_train, 8*L_train}, reporting accuracy on the
last-position token and all-position average.

Notes:
    - random-guess accuracy = 1/n!  (n=3 -> 16.67%, n=5 -> 0.83%)
    - chunk_size must divide each eval length (we pick powers of 2 of chunk_size).
"""

import sys
from dataclasses import dataclass, field
from itertools import permutations
from math import factorial
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from rnn.non_linear_linear_rnn import (
    NonLinearLinearRNNConfig,
    NonLinearLinearRNNLM,
)


# ---------------------------------------------------------------------------
# Reference M²RNN implementation (for sanity-check comparison).
#
# Lives at /gpfs/data/.../state_tracking/models/rnn/, vendored 1:1 from
# open-lm-engine/accelerated-model-architectures (the M²RNN paper authors'
# reference impl). We import on demand so users without that path don't see
# a hard import error.
# ---------------------------------------------------------------------------

_REF_ROOT = Path("/gpfs/data/ranganathlab/Jatin/model_architecture_stuff")


def _import_ref_rnn_model():
    """Import the reference RNNModel class from the vendored M²RNN baseline."""
    if str(_REF_ROOT) not in sys.path:
        sys.path.insert(0, str(_REF_ROOT))
    from state_tracking.models.rnn.model import RNNModel  # type: ignore
    return RNNModel


# ---------------------------------------------------------------------------
# S_n Cayley table
# ---------------------------------------------------------------------------


def _build_sn_cayley(n: int):
    """Enumerate S_n and build its multiplication table.

    Elements are tuples representing permutations of (0, ..., n-1).
    Composition convention: (a . b)[i] = a[b[i]], i.e. b applied first, then a.
    With this convention, if we track `running` left-to-right and do
    `running = running . g_t` at each step, `running` equals the product
    g_1 . g_2 . ... . g_t.
    """
    perms = sorted(permutations(range(n)))  # canonical index order
    size = len(perms)
    idx = {p: i for i, p in enumerate(perms)}
    table = [[0] * size for _ in range(size)]
    for i, a in enumerate(perms):
        for j, b in enumerate(perms):
            c = tuple(a[b[k]] for k in range(n))
            table[i][j] = idx[c]
    return perms, table


def make_batch(B: int, L: int, vocab_size: int, cayley_t: torch.Tensor,
                identity_idx: int, device: str):
    """Return (input_ids, target_ids) of shape (B, L), both in {0..vocab_size-1}.

    input_ids[b, t]  ~ Uniform(S_n)
    target_ids[b, t] = running product of input_ids[b, 0..t] inclusive.
    """
    g = torch.randint(0, vocab_size, (B, L), device=device)
    table = cayley_t.to(device)
    targets = torch.empty_like(g)
    running = torch.full((B,), identity_idx, dtype=torch.long, device=device)
    for t in range(L):
        # running <- running . g_t  (row = running, col = g_t)
        running = table[running, g[:, t]]
        targets[:, t] = running
    return g, targets


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class SnConfig:
    # which symmetric group: vocab_size = n!
    # The M²RNN paper (Mishra et al. 2026) only validates state tracking on
    # S_3 (Figure 3), training at L=128 and evaluating up to L=512 — so we
    # default to that exact setup as a sanity check that our M²RNN impl is
    # correct. Bump to 5 (S_5) to attempt the harder non-solvable case.
    group_n: int = 3

    # lengths (chunk_size must divide each)
    # Short sequences for fast iteration: each m2rnn forward is sequential
    # over L (L tanh @ matmuls per token), so cutting L=128 -> L=16 gives
    # ~8x speedup. We keep some longer eval lengths to still see whether
    # the model length-extrapolates beyond the training horizon.
    L_train: int = 16
    L_eval: tuple = (16, 32, 64, 128)

    # training
    batch_size: int = 256
    steps: int = 5000
    lr: float = 1e-4
    seed: int = 0
    log_every: int = 100

    # Match the reference (state_tracking/config/model/rnn.yaml) which is the
    # M²RNN paper / DeltaProduct paper baseline: hidden=384, heads=12,
    # layers=1, K=V=32. With this, dim == num_heads * head_v_dim == 384 (the
    # standard multi-head attention convention). Capacity is apples-to-apples
    # with the ref impl so any remaining gap is architectural (no SwiGLU MLP,
    # `full_block` extras, sigmoid on f, identity W init) rather than scale.
    dim: int = 384
    num_heads: int = 12
    head_k_dim: int = 32
    head_v_dim: int = 32
    layers: int = 1
    chunk_size: int = 4

    # For group state tracking we do NOT want the state to decay: the entire
    # "memory" is the current group element and we want to carry it forward
    # faithfully. Let the model learn stronger decay if it needs to.
    decay_init: str = "mild"
    # Short conv gives a local-window view (kernel 4) that the model can use
    # to compute within-chunk partial products cheaply before the boundary.
    use_short_conv: bool = True
    # Allow the per-token Householder factor (I - beta k k^T) to have negative
    # eigenvalues (beta in (0, 2) instead of (0, 1)). Mandatory for S_n word
    # problems (Grazzi et al. 2024); see module docstring.
    allow_neg_eigval: bool = True

    # Top-level architecture: "linear_chunks" (delta-rule WY kernel + boundary
    # non-linearity) or "m2rnn" (pure M²RNN per-token recurrence). When
    # "m2rnn", `variants` and `chunk_size` are ignored.
    architecture: str = "linear_chunks"

    # boundary non-linearities to sweep — only used when architecture="linear_chunks"
    # variants: tuple = ("identity", "rmsnorm", "gru", "gru_input", "m2rnn")
    variants: tuple = ("identity", "rmsnorm", "tanh_res", "gru")

    # When True, ALSO run the reference (vendored) M²RNN impl from
    # /gpfs/data/.../state_tracking/models/rnn/ side-by-side, as a sanity
    # check that the benchmark itself is solvable. Uses the reference's
    # paper-matching config (hidden=384, num_heads=12, layers=1,
    # intermediate=1024, SwiGLU MLP, no full_block hacks).
    run_reference_m2rnn: bool = True
    # When True, skip the in-house variants entirely and run ONLY the
    # reference impl. Useful for verifying the benchmark is solvable.
    run_reference_only: bool = False
    # Default ref dims match ours (so the comparison is apples-to-apples on
    # capacity). intermediate_size uses 3x hidden (SwiGLU's typical ratio).

    ref_hidden_size: int = 384
    ref_intermediate_size: int = 1024
    ref_num_heads: int = 12

    # ref_hidden_size: int = 128
    # ref_intermediate_size: int = 384
    # ref_num_heads: int = 2

    ref_n_layers: int = 1
    ref_key_head_dim: int = 32
    ref_value_head_dim: int = 32
    # "triton" (fused CUDA kernel from accelerated-model-architectures, much
    # faster on A100) or "torch" (pure-PyTorch reference, CPU-friendly).
    # Correctness is equivalent.
    ref_backend: str = "triton"


def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# generic train + eval loop (works for any (B, L) -> (B, L, vocab) model)
# ---------------------------------------------------------------------------


def _train_and_eval(model, tag: str, cfg: SnConfig, cayley_t: torch.Tensor,
                    identity_idx: int, vocab_size: int, device: str,
                    chunk_divisor: int = 1):
    """Generic training + length-extrapolation eval loop.

    chunk_divisor is the value (typically 1, or `cfg.chunk_size` for the
    linear_chunks architecture) that all eval lengths must be divisible by.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    desc = f"S{cfg.group_n} [{tag:<14s}]"
    bar = tqdm(range(cfg.steps), desc=desc, mininterval=0.5)

    model.train()
    for step in bar:
        input_ids, target_ids = make_batch(
            cfg.batch_size, cfg.L_train, vocab_size, cayley_t, identity_idx, device
        )
        logits = model(input_ids)                                       # (B, L, vocab)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size), target_ids.reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            acc = (logits.argmax(-1) == target_ids).float().mean().item()
        bar.set_postfix_str(f"loss={loss.item():.3f} acc={acc:.2%} grad_norm={grad_norm:.3f}")

    model.eval()
    eval_B = max(cfg.batch_size, 128)
    all_acc, last_acc = {}, {}
    with torch.no_grad():
        for L in cfg.L_eval:
            assert L % chunk_divisor == 0, f"L={L} not divisible by chunk_divisor={chunk_divisor}"
            input_ids, target_ids = make_batch(
                eval_B, L, vocab_size, cayley_t, identity_idx, device
            )
            logits = model(input_ids)
            preds = logits.argmax(-1)
            all_acc[L] = (preds == target_ids).float().mean().item()
            last_acc[L] = (preds[:, -1] == target_ids[:, -1]).float().mean().item()
    return {"all_acc": all_acc, "last_acc": last_acc}


# ---------------------------------------------------------------------------
# run one variant of OUR impl (linear_chunks or m2rnn)
# ---------------------------------------------------------------------------


def run_variant(boundary_nonlin: str, cfg: SnConfig, cayley_t: torch.Tensor,
                identity_idx: int, vocab_size: int, device: str):
    torch.manual_seed(cfg.seed)

    model_config = NonLinearLinearRNNConfig(
        dim=cfg.dim,
        num_heads=cfg.num_heads,
        head_k_dim=cfg.head_k_dim,
        head_v_dim=cfg.head_v_dim,
        layers=cfg.layers,
        vocab_size=vocab_size,
        chunk_size=cfg.chunk_size,
        boundary_nonlin=boundary_nonlin,
        use_short_conv=cfg.use_short_conv,
        decay_init=cfg.decay_init,
        allow_neg_eigval=cfg.allow_neg_eigval,
        architecture=cfg.architecture,
    )
    model = NonLinearLinearRNNLM(model_config).to(device)
    tag = cfg.architecture if cfg.architecture == "m2rnn" else boundary_nonlin
    chunk_div = 1 if cfg.architecture == "m2rnn" else cfg.chunk_size
    return _train_and_eval(model, tag, cfg, cayley_t, identity_idx, vocab_size,
                            device, chunk_divisor=chunk_div)


# ---------------------------------------------------------------------------
# run reference M²RNN impl (vendored from accelerated-model-architectures)
# ---------------------------------------------------------------------------


def run_ref_m2rnn(cfg: SnConfig, cayley_t: torch.Tensor, identity_idx: int,
                  vocab_size: int, device: str):
    torch.manual_seed(cfg.seed)
    RNNModel = _import_ref_rnn_model()
    # Match the reference's paper config (config/model/rnn.yaml).
    model = RNNModel(
        vocab_size=vocab_size,
        hidden_size=cfg.ref_hidden_size,
        intermediate_size=cfg.ref_intermediate_size,
        n_layers=cfg.ref_n_layers,
        num_heads=cfg.ref_num_heads,
        key_head_dim=cfg.ref_key_head_dim,
        value_head_dim=cfg.ref_value_head_dim,
        backend=cfg.ref_backend,
        gradient_clipping=None,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"  [ref m2rnn] hidden={cfg.ref_hidden_size} heads={cfg.ref_num_heads} "
        f"layers={cfg.ref_n_layers} K={cfg.ref_key_head_dim} V={cfg.ref_value_head_dim} "
        f"intermediate={cfg.ref_intermediate_size}  backend={cfg.ref_backend}  "
        f"params={n_params/1e6:.2f}M"
    )
    return _train_and_eval(model, "ref_m2rnn", cfg, cayley_t, identity_idx,
                            vocab_size, device, chunk_divisor=1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    cfg = SnConfig()
    device = _pick_device()

    # build Cayley table for S_n
    perms, cayley = _build_sn_cayley(cfg.group_n)
    vocab_size = factorial(cfg.group_n)
    assert len(perms) == vocab_size
    identity_idx = perms.index(tuple(range(cfg.group_n)))
    cayley_t = torch.tensor(cayley, dtype=torch.long)

    print(f"device={device}")
    print(
        f"S_{cfg.group_n} state-tracking | |S_{cfg.group_n}|={vocab_size}  "
        f"train L={cfg.L_train}  eval L={list(cfg.L_eval)}"
    )
    if cfg.architecture == "m2rnn":
        print(
            f"architecture=m2rnn (pure non-linear RNN per token; chunk_size and "
            f"boundary_nonlin are ignored)"
        )
        print(
            f"dim={cfg.dim}  heads={cfg.num_heads}  layers={cfg.layers}  "
            f"head_k_dim={cfg.head_k_dim}  head_v_dim={cfg.head_v_dim}"
        )
    else:
        print(
            f"architecture=linear_chunks  dim={cfg.dim}  heads={cfg.num_heads}  "
            f"layers={cfg.layers}  chunk_size={cfg.chunk_size}  "
            f"(train chunks={cfg.L_train // cfg.chunk_size})  "
            f"allow_neg_eigval={cfg.allow_neg_eigval}"
        )
    print(f"random-guess accuracy: {1 / vocab_size:.2%}\n")

    # When architecture=m2rnn, boundary_nonlin is unused; we still loop once
    # so the existing report machinery works.
    results = {}
    if not cfg.run_reference_only:
        variants_to_run = cfg.variants if cfg.architecture == "linear_chunks" else ("m2rnn",)
        for kind in variants_to_run:
            results[kind] = run_variant(
                kind, cfg, cayley_t, identity_idx, vocab_size, device
            )

    # Sanity-check: also run the reference (vendored) M²RNN impl on the same
    # task. If this converges and ours doesn't, the bug is in our impl, not
    # the benchmark.
    if cfg.run_reference_m2rnn or cfg.run_reference_only:
        if results:
            print()
        results["ref_m2rnn"] = run_ref_m2rnn(
            cfg, cayley_t, identity_idx, vocab_size, device
        )

    # Pretty-print two tables: per-token accuracy and last-position accuracy.
    def _print_table(title, key):
        print(f"\n=== S_{cfg.group_n} {title} ===")
        header = f"  {'variant':<14s} " + "".join(f"{f'L={L}':>10s} " for L in cfg.L_eval)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for kind, r in results.items():
            row = f"  {kind:<14s} " + "".join(f"{r[key][L]:>10.2%} " for L in cfg.L_eval)
            print(row)

    _print_table("all-position accuracy", "all_acc")
    _print_table("last-position accuracy", "last_acc")


if __name__ == "__main__":
    main()
