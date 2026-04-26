"""
Dyck-2 stack-top length-generalization benchmark.

Task:
    Vocabulary (input):  4 tokens
        0 = '('   open round
        1 = ')'   close round
        2 = '['   open square
        3 = ']'   close square

    We sample only WELL-FORMED Dyck-2 prefixes:
        - if stack is empty:    push '(' or '[' uniformly at random
        - else:                 50% push (bracket type uniform), 50% pop top

    Target at each position is the bracket type currently on top of the stack
    (3 classes):
        0 = empty
        1 = '('
        2 = '['

Why Dyck-2 is genuinely non-linear (and unlike S_n / saturated-counter):
    Dyck-2 is the canonical strictly-context-free language. Tracking the
    stack TOP across arbitrary depth requires remembering the entire stack:
    after popping, the new top is whatever was second-from-top, which is
    only available if we also remember third-from-top, etc.

    The number of distinct stacks of depth d is 2^d. A linear RNN with
    state-dim N can simulate any DFA with up to ~N states, but Dyck-2's
    "DFA" has 2^d states for depth d -> exponential in depth, so any
    fixed-dim linear model fails past some critical depth.

    Concretely, our 32x32 per-head state holds ~10 bits of stack info before
    saturating. So:
        - L <= ~32  (avg depth ~4):  linear model should still cope.
        - L >= ~128 (avg depth ~8+): linear model should start to fail
                                     at deep positions, while non-linear
                                     boundaries (gru, tanh_res) and pure
                                     M2RNN should keep working.

Setup mirrors non_linear_linear_rnn_satcount.py: same model, same training
loop, same sweep over boundary nonlinearities + reference M2RNN side-by-side.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from rnn.non_linear_linear_rnn import (
    NonLinearLinearRNNConfig,
    NonLinearLinearRNNLM,
)


# ---------------------------------------------------------------------------
# Reference M2RNN (vendored, same as in the S_n script).
# ---------------------------------------------------------------------------

_REF_ROOT = Path("/gpfs/data/ranganathlab/Jatin/model_architecture_stuff")


def _import_ref_rnn_model():
    if str(_REF_ROOT) not in sys.path:
        sys.path.insert(0, str(_REF_ROOT))
    from state_tracking.models.rnn.model import RNNModel  # type: ignore
    return RNNModel


# ---------------------------------------------------------------------------
# data generator
# ---------------------------------------------------------------------------

OPEN_PAREN, CLOSE_PAREN, OPEN_SQUARE, CLOSE_SQUARE = 0, 1, 2, 3
INPUT_VOCAB = 4

# Stack-top labels:
EMPTY_TOP, ROUND_TOP, SQUARE_TOP = 0, 1, 2
NUM_CLASSES = 3


def make_dyck2_batch(B: int, L: int, device: str):
    """Generate (B, L) batches of BALANCED, DEEP Dyck-2 strings + stack-top.

    "Balanced" means equal pushes and pops, ending at depth 0. "Deep" means
    we use a time-decreasing push probability `(L - t) / L` so the depth
    profile is roughly triangular: ramps up early, peaks at t = L/2 with
    expected depth ~L/4, then ramps down. For L=64 peak depth is ~16 -> 2^16
    distinct stack configurations, well past the linear-kernel's bits-of-
    state capacity (~9 bits in our tiny config). This is what makes the
    linear ceiling fail catastrophically rather than gracefully.

    Constraints (same as before):
        - if depth == 0:              must push (can't pop an empty stack)
        - if depth == L - t:          must pop  (no slack left to balance)
        - otherwise:                  push w.p. (L - t) / L, else pop

    L must be even.
    """
    assert L % 2 == 0, f"balanced Dyck-2 requires even L, got L={L}"
    MAX_DEPTH = L // 2 + 1
    stack = torch.zeros(B, MAX_DEPTH, dtype=torch.long, device=device)
    depth = torch.zeros(B, dtype=torch.long, device=device)

    x = torch.empty(B, L, dtype=torch.long, device=device)
    targets = torch.empty(B, L, dtype=torch.long, device=device)

    for t in range(L):
        remaining = L - t
        is_empty = depth == 0
        must_pop = depth == remaining
        push_prob = remaining / L
        u = torch.rand(B, device=device)
        push = is_empty | (~must_pop & (u < push_prob))

        bt = (torch.rand(B, device=device) < 0.5).long()  # 0 -> round, 1 -> square

        idx = (depth - 1).clamp(min=0)
        cur_top = stack.gather(1, idx.unsqueeze(1)).squeeze(1)

        push_token = torch.where(bt == 0, torch.full_like(bt, OPEN_PAREN),
                                 torch.full_like(bt, OPEN_SQUARE))
        pop_token = torch.where(cur_top == ROUND_TOP,
                                torch.full_like(cur_top, CLOSE_PAREN),
                                torch.full_like(cur_top, CLOSE_SQUARE))
        tok = torch.where(push, push_token, pop_token)
        x[:, t] = tok

        new_entry = bt + 1
        push_idx = depth.clamp(max=MAX_DEPTH - 1).unsqueeze(1)
        write_vals = torch.where(push, new_entry, stack.gather(1, push_idx).squeeze(1))
        stack.scatter_(1, push_idx, write_vals.unsqueeze(1))
        delta = torch.where(push, torch.ones_like(depth), -torch.ones_like(depth))
        depth = (depth + delta).clamp(min=0)

        new_idx = (depth - 1).clamp(min=0)
        new_top = stack.gather(1, new_idx.unsqueeze(1)).squeeze(1)
        new_top = torch.where(depth == 0, torch.full_like(new_top, EMPTY_TOP), new_top)
        targets[:, t] = new_top

    return x, targets


def _is_pop_token(x: torch.Tensor) -> torch.Tensor:
    """Mask of positions whose input is a closing bracket (i.e. a pop op)."""
    return (x == CLOSE_PAREN) | (x == CLOSE_SQUARE)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class Dyck2Config:
    # lengths (chunk_size must divide each); also must be EVEN for balanced
    # Dyck-2 generation.
    L_train: int = 64
    L_eval: tuple = (64, 128, 256, 512)

    batch_size: int = 256
    steps: int = 2000
    lr: float = 1e-4
    seed: int = 0

    # Early-stop training when training acc has been >= this threshold for
    # `early_stop_window` consecutive steps. Set to None to disable.
    early_stop_acc: float | None = 0.95
    early_stop_window: int = 20

    # Tiny model: ~32k params total. With state-dim 16x16=256 per head and
    # 2 heads, the linear kernel can simulate at most ~2^9 distinct stack
    # configurations, so beyond depth ~9 it must lose information. This
    # exposes the linear ceiling clearly and lets non-linear boundaries
    # actually pull ahead.
    dim: int = 64
    num_heads: int = 2
    head_k_dim: int = 16
    head_v_dim: int = 16
    layers: int = 1
    chunk_size: int = 4

    decay_init: str = "mild"
    # Disable the kernel-4 short conv on q/k/v: with it, the model gets a
    # free 4-token lookahead that handles shallow stack regions (depth 1-2)
    # without using the recurrent state. Killing it forces ALL predictions
    # through the linear kernel, exposing its capacity ceiling more cleanly.
    use_short_conv: bool = False
    allow_neg_eigval: bool = True

    architecture: str = "linear_chunks"
    variants: tuple = ("identity", "rmsnorm", "tanh_res", "gru")

    run_reference_m2rnn: bool = True
    run_reference_only: bool = False
    # Match our tiny config for an apples-to-apples comparison.
    ref_hidden_size: int = 64
    ref_intermediate_size: int = 128
    ref_num_heads: int = 2
    ref_n_layers: int = 1
    ref_key_head_dim: int = 16
    ref_value_head_dim: int = 16
    ref_backend: str = "triton"


def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# train + eval
# ---------------------------------------------------------------------------


def _train_and_eval(model, tag: str, cfg: Dyck2Config, num_classes: int,
                    device: str, chunk_divisor: int = 1):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    desc = f"dyck2 [{tag:<14s}]"
    bar = tqdm(range(cfg.steps), desc=desc, mininterval=0.5)

    model.train()
    consec_above = 0
    for step in bar:
        input_ids, target_ids = make_dyck2_batch(
            cfg.batch_size, cfg.L_train, device
        )
        logits = model(input_ids)
        ce_logits = logits[..., :num_classes]
        loss = F.cross_entropy(
            ce_logits.reshape(-1, num_classes), target_ids.reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            preds = ce_logits.argmax(-1)
            correct = preds == target_ids
            acc = correct.float().mean().item()
            pop_mask = _is_pop_token(input_ids)
            pop_acc = (
                (correct & pop_mask).sum().float() / pop_mask.sum().clamp(min=1).float()
            ).item()
        bar.set_postfix_str(
            f"loss={loss.item():.3f} acc={acc:.2%} pop_acc={pop_acc:.2%} "
            f"grad_norm={grad_norm:.3f}"
        )
        if cfg.early_stop_acc is not None:
            # Early-stop on pop_acc (the metric that actually requires stack
            # tracking), not all-position acc, since the latter is dominated
            # by trivially-predictable post-push positions.
            consec_above = consec_above + 1 if pop_acc >= cfg.early_stop_acc else 0
            if consec_above >= cfg.early_stop_window:
                bar.set_postfix_str(
                    f"loss={loss.item():.3f} acc={acc:.2%} "
                    f"pop_acc={pop_acc:.2%} early-stop@step={step}"
                )
                bar.close()
                break

    model.eval()
    eval_B = max(cfg.batch_size, 128)
    all_acc, pop_acc, last_acc = {}, {}, {}
    with torch.no_grad():
        for L in cfg.L_eval:
            assert L % chunk_divisor == 0, (
                f"L={L} not divisible by chunk_divisor={chunk_divisor}"
            )
            input_ids, target_ids = make_dyck2_batch(eval_B, L, device)
            logits = model(input_ids)
            preds = logits[..., :num_classes].argmax(-1)
            correct = preds == target_ids
            all_acc[L] = correct.float().mean().item()
            pop_mask = _is_pop_token(input_ids)
            pop_acc[L] = (
                (correct & pop_mask).sum().float() / pop_mask.sum().clamp(min=1).float()
            ).item()
            last_acc[L] = (preds[:, -1] == target_ids[:, -1]).float().mean().item()
    return {"all_acc": all_acc, "pop_acc": pop_acc, "last_acc": last_acc}


# ---------------------------------------------------------------------------
# variant runners
# ---------------------------------------------------------------------------


def run_variant(boundary_nonlin: str, cfg: Dyck2Config, num_classes: int,
                vocab_size: int, device: str):
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
    return _train_and_eval(model, tag, cfg, num_classes, device,
                           chunk_divisor=chunk_div)


def run_ref_m2rnn(cfg: Dyck2Config, num_classes: int, vocab_size: int,
                  device: str):
    torch.manual_seed(cfg.seed)
    RNNModel = _import_ref_rnn_model()
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
    return _train_and_eval(model, "ref_m2rnn", cfg, num_classes, device,
                           chunk_divisor=1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    cfg = Dyck2Config()
    device = _pick_device()

    num_classes = NUM_CLASSES
    vocab_size = max(INPUT_VOCAB, num_classes)

    print(f"device={device}")
    print(
        f"Dyck-2 stack-top (balanced) | input_vocab={INPUT_VOCAB}  "
        f"out_classes={num_classes}  train L={cfg.L_train}  "
        f"eval L={list(cfg.L_eval)}"
    )
    if cfg.architecture == "m2rnn":
        print(
            f"architecture=m2rnn  dim={cfg.dim}  heads={cfg.num_heads}  "
            f"layers={cfg.layers}  K={cfg.head_k_dim}  V={cfg.head_v_dim}"
        )
    else:
        print(
            f"architecture=linear_chunks  dim={cfg.dim}  heads={cfg.num_heads}  "
            f"layers={cfg.layers}  chunk_size={cfg.chunk_size}  "
            f"allow_neg_eigval={cfg.allow_neg_eigval}"
        )
    print(f"random-guess accuracy: {1 / num_classes:.2%}\n")

    results = {}
    if not cfg.run_reference_only:
        variants_to_run = (
            cfg.variants if cfg.architecture == "linear_chunks" else ("m2rnn",)
        )
        for kind in variants_to_run:
            results[kind] = run_variant(kind, cfg, num_classes, vocab_size, device)

    if cfg.run_reference_m2rnn or cfg.run_reference_only:
        if results:
            print()
        results["ref_m2rnn"] = run_ref_m2rnn(cfg, num_classes, vocab_size, device)

    def _print_table(title, key):
        print(f"\n=== Dyck-2 {title} ===")
        header = f"  {'variant':<14s} " + "".join(f"{f'L={L}':>10s} " for L in cfg.L_eval)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for kind, r in results.items():
            row = (
                f"  {kind:<14s} "
                + "".join(f"{r[key][L]:>10.2%} " for L in cfg.L_eval)
            )
            print(row)

    _print_table("all-position accuracy", "all_acc")
    _print_table("post-pop accuracy (real stack-tracking signal)", "pop_acc")
    _print_table("last-position accuracy", "last_acc")


if __name__ == "__main__":
    main()
