"""
S_3 state-tracking length-generalization benchmark.

Task:
    input_ids[t]  = random element g_t in S_3            (vocab = 6)
    target_ids[t] = cumulative product g_1 . g_2 . ... . g_t  (vocab = 6)

S_3 is the smallest non-abelian group. Composing its elements cannot be done
by a bounded-depth parallel circuit and, in practice, pure linear RNNs struggle
to length-generalize the running product. A non-linear RNN (or our hybrid with
a non-linear boundary f(S)) should be able to extrapolate past the training
length because the group's finite-automaton transition function is expressible
exactly in closed form by a non-linear update.

We train on sequences of length L_train and evaluate on lengths
{L_train, 2*L_train, 4*L_train, 8*L_train}, reporting accuracy on the
last-position token and all-position average.

Expected outcome (the point of the benchmark):
    - identity (pure linear RNN boundary): solves L_train but accuracy degrades
      fast at longer lengths.
    - gru (non-linear boundary): solves L_train AND holds accuracy at longer
      lengths, because the boundary can implement the group-action transition.

Notes:
    - random-guess accuracy = 1/6 ~ 16.67%
    - chunk_size must divide each eval length (we pick powers of 2 of chunk_size).
"""

from dataclasses import dataclass, field
from itertools import permutations

import torch
import torch.nn.functional as F
from tqdm import tqdm

from rnn.non_linear_linear_rnn import (
    NonLinearLinearRNNConfig,
    NonLinearLinearRNNLM,
)


# ---------------------------------------------------------------------------
# S_3 Cayley table
# ---------------------------------------------------------------------------


def _build_s3_cayley():
    """Enumerate S_3 and build its multiplication table.

    Elements are tuples representing permutations of (0, 1, 2).
    Composition convention: (a . b)[i] = a[b[i]], i.e. b applied first, then a.
    With this convention, if we track `running` left-to-right and do
    `running = running . g_t` at each step, `running` equals the product
    g_1 . g_2 . ... . g_t.
    """
    perms = sorted(permutations(range(3)))  # canonical index order
    n = len(perms)
    idx = {p: i for i, p in enumerate(perms)}
    table = [[0] * n for _ in range(n)]
    for i, a in enumerate(perms):
        for j, b in enumerate(perms):
            c = tuple(a[b[k]] for k in range(3))
            table[i][j] = idx[c]
    return perms, table


PERMS, CAYLEY = _build_s3_cayley()
# IDENTITY_IDX is whatever index the tuple (0,1,2) lands at under sorted order.
IDENTITY_IDX = PERMS.index((0, 1, 2))
CAYLEY_T = torch.tensor(CAYLEY, dtype=torch.long)        # (6, 6)


def make_batch(B: int, L: int, device: str):
    """Return (input_ids, target_ids) of shape (B, L), both in {0..5}.

    input_ids[b, t]  ~ Uniform(S_3)
    target_ids[b, t] = running product of input_ids[b, 0..t] inclusive.
    """
    g = torch.randint(0, 6, (B, L), device=device)
    table = CAYLEY_T.to(device)
    targets = torch.empty_like(g)
    running = torch.full((B,), IDENTITY_IDX, dtype=torch.long, device=device)
    for t in range(L):
        # running <- running . g_t  (row = running, col = g_t)
        running = table[running, g[:, t]]
        targets[:, t] = running
    return g, targets


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class S3Config:
    # lengths (chunk_size must divide each)
    L_train: int = 64
    L_eval: tuple = (64, 128, 256, 512)

    # training
    batch_size: int = 256
    steps: int = 5000
    lr: float = 1e-4
    seed: int = 0
    log_every: int = 100

    # model -- chunk_size=4 gives L/4 = 16 boundary applications per training
    # sequence, which gives the boundary nonlinearity enough gradient signal to
    # actually learn the group transition function.
    dim: int = 96
    num_heads: int = 2
    head_k_dim: int = 24
    head_v_dim: int = 24
    layers: int = 2
    chunk_size: int = 4
    vocab_size: int = 6

    # For group state tracking we do NOT want the state to decay: the entire
    # "memory" is the current group element and we want to carry it forward
    # faithfully. Let the model learn stronger decay if it needs to.
    decay_init: str = "mild"
    # Short conv gives a local-window view (kernel 4) that the model can use
    # to compute within-chunk partial products cheaply before the boundary.
    use_short_conv: bool = True
    # Allow the per-token Householder factor (I - beta k k^T) to have negative
    # eigenvalues (beta in (0, 2) instead of (0, 1)). This is strictly
    # necessary for *any* non-trivial group word problem: with beta in (0, 1)
    # the state-transition eigenvalues are all in (0, 1) and a finite-precision
    # linear RNN provably cannot even solve parity (Grazzi et al. 2024,
    # "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues",
    # arXiv:2411.12537). This is a prerequisite for S_3; solving S_3 in one
    # layer additionally needs >=2 Householders per token (DeltaProduct).
    allow_neg_eigval: bool = True

    # which boundary non-linearities to sweep
    variants: tuple = ("identity", "rmsnorm", "gru")


def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# run one variant
# ---------------------------------------------------------------------------


def run_variant(boundary_nonlin: str, cfg: S3Config, device: str):
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
        decay_init=cfg.decay_init,
        allow_neg_eigval=cfg.allow_neg_eigval,
    )
    model = NonLinearLinearRNNLM(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    desc = f"S3 [{boundary_nonlin:<9s}]"
    bar = tqdm(range(cfg.steps), desc=desc, mininterval=0.5)

    model.train()
    for step in bar:
        input_ids, target_ids = make_batch(cfg.batch_size, cfg.L_train, device)
        logits = model(input_ids)                                       # (B, L, 6)
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), target_ids.reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # if (step + 1) % cfg.log_every == 0 or step == cfg.steps - 1:
        with torch.no_grad():
            acc = (logits.argmax(-1) == target_ids).float().mean().item()
        bar.set_postfix_str(f"loss={loss.item():.3f} acc={acc:.2%} grad_norm={grad_norm:.3f}")

    # Eval on each length; also measure the last-position accuracy specifically
    # because that's the hardest position (requires the full running product).
    model.eval()
    eval_B = max(cfg.batch_size, 128)
    all_acc = {}
    last_acc = {}
    with torch.no_grad():
        for L in cfg.L_eval:
            assert L % cfg.chunk_size == 0, f"L={L} not divisible by chunk_size={cfg.chunk_size}"
            input_ids, target_ids = make_batch(eval_B, L, device)
            logits = model(input_ids)
            preds = logits.argmax(-1)
            all_acc[L] = (preds == target_ids).float().mean().item()
            last_acc[L] = (preds[:, -1] == target_ids[:, -1]).float().mean().item()

    return {"all_acc": all_acc, "last_acc": last_acc}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    cfg = S3Config()
    device = _pick_device()
    print(f"device={device}")
    print(f"S_3 state-tracking | train L={cfg.L_train}  eval L={list(cfg.L_eval)}")
    print(
        f"dim={cfg.dim}  heads={cfg.num_heads}  layers={cfg.layers}  "
        f"chunk_size={cfg.chunk_size}  (train chunks={cfg.L_train // cfg.chunk_size})  "
        f"allow_neg_eigval={cfg.allow_neg_eigval}"
    )
    print(f"random-guess accuracy: {1 / 6:.2%}\n")

    results = {}
    for kind in cfg.variants:
        results[kind] = run_variant(kind, cfg, device)

    # Pretty-print two tables: per-token accuracy and last-position accuracy.
    def _print_table(title, key):
        print(f"\n=== S_3 {title} ===")
        header = f"  {'variant':<10s} " + "".join(f"{f'L={L}':>10s} " for L in cfg.L_eval)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for kind, r in results.items():
            row = f"  {kind:<10s} " + "".join(f"{r[key][L]:>10.2%} " for L in cfg.L_eval)
            print(row)

    _print_table("all-position accuracy", "all_acc")
    _print_table("last-position accuracy", "last_acc")


if __name__ == "__main__":
    main()
