"""
Saturated counter length-generalization benchmark.

Task:
    Vocabulary (input):  3 tokens
        0 = '+'   increment counter (saturated at K)
        1 = '-'   decrement counter (saturated at 0)
        2 = '='   query: emit current counter, no state change

    State:  integer in {0, 1, ..., K}, starts at 0.
    Transitions:
        '+'  ->  state = min(state + 1, K)
        '-'  ->  state = max(state - 1, 0)
        '='  ->  state unchanged, target = state

    Loss is cross-entropy at '=' positions only; non-'=' positions use
    ignore_index = -100. Accuracy is reported on '=' positions only
    (last '=' and all '=').

Why this task is genuinely non-linear:
    The min/max saturation cannot be expressed by a linear recurrence
        S_t = A(x_t) S_{t-1} + b(x_t)
    Without saturation the state would either drift unboundedly or wrap
    modulo K+1; with saturation the evolution depends on the *value*
    of the state itself ('+' is identity at state=K, no-op; '-' is
    identity at state=0, no-op). State-dependent transitions => non-linear.

    Predicted ordering:
        identity (linear delta-rule + no boundary nonlin) should fail.
        tanh_res / gru / m2rnn-architecture should succeed (tanh provides
            saturation natively).
    If we see this ordering flip relative to the S_n result, we've
    isolated *what kind of task non-linearity helps with*.

Setup mirrors non_linear_linear_rnn_s3.py: same model, same training loop,
same sweep over boundary nonlinearities + reference M²RNN side-by-side.
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
# Reference M²RNN (vendored, same as in the S_n script).
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

PLUS, MINUS, EQ = 0, 1, 2
INPUT_VOCAB = 3
IGNORE = -100


def make_satcount_batch(B: int, L: int, K: int, eq_prob: float, device: str):
    """Generate a (B, L) batch and ground-truth saturated-counter targets.

    eq_prob is the marginal probability that a token is '='. Remaining
    probability is split equally between '+' and '-'.
    """
    p_pm = (1.0 - eq_prob) / 2.0
    probs = torch.tensor([p_pm, p_pm, eq_prob], device=device)
    x = torch.multinomial(probs, B * L, replacement=True).view(B, L)

    state = torch.zeros(B, dtype=torch.long, device=device)
    targets = torch.full((B, L), IGNORE, dtype=torch.long, device=device)
    K_t = torch.tensor(K, device=device)
    zero_t = torch.tensor(0, device=device)
    for t in range(L):
        tok = x[:, t]
        plus_mask = tok == PLUS
        minus_mask = tok == MINUS
        eq_mask = tok == EQ
        state = torch.where(plus_mask, torch.minimum(state + 1, K_t), state)
        state = torch.where(minus_mask, torch.maximum(state - 1, zero_t), state)
        targets[:, t] = torch.where(eq_mask, state, torch.tensor(IGNORE, device=device))
    return x, targets


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class SatCountConfig:
    # K=2 means the counter saturates after just 2 consecutive +/- (common
    # in 16 tokens), so min/max actually trigger during training. With K=8
    # the random walk almost never reaches the boundary at L=16 and the
    # task degenerates to "running signed sum", which is linear and
    # trivially solved by the identity boundary.
    K: int = 2
    eq_prob: float = 0.25

    # lengths (chunk_size must divide each). With eq_prob=0.25 and L=16
    # we get ~4 labeled positions per sequence on average — a bit sparse
    # but matches the S_n script's training horizon for a clean comparison.
    L_train: int = 16
    L_eval: tuple = (16, 32, 64, 128)

    # training
    batch_size: int = 256
    steps: int = 5000
    lr: float = 1e-4
    seed: int = 0

    # model — same as S_n script for apples-to-apples
    dim: int = 384
    num_heads: int = 12
    head_k_dim: int = 32
    head_v_dim: int = 32
    layers: int = 1
    chunk_size: int = 4

    decay_init: str = "mild"
    use_short_conv: bool = True
    allow_neg_eigval: bool = True

    architecture: str = "linear_chunks"
    variants: tuple = ("identity", "rmsnorm", "tanh_res", "gru")

    run_reference_m2rnn: bool = True
    run_reference_only: bool = False
    ref_hidden_size: int = 384
    ref_intermediate_size: int = 1024
    ref_num_heads: int = 12
    ref_n_layers: int = 1
    ref_key_head_dim: int = 32
    ref_value_head_dim: int = 32
    ref_backend: str = "triton"


def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# train + eval (mirrors the S_n script, but loss/acc only on '=' positions)
# ---------------------------------------------------------------------------


def _eq_acc(logits, targets, num_classes):
    """Accuracy at positions where targets != IGNORE (i.e. '=' positions)."""
    mask = targets != IGNORE
    if mask.sum() == 0:
        return float("nan"), float("nan")
    preds = logits[..., :num_classes].argmax(-1)
    correct = (preds == targets) & mask
    all_acc = correct.sum().float() / mask.sum().float()
    # last '=' per row
    L = targets.shape[1]
    pos = torch.arange(L, device=targets.device).expand_as(targets)
    last_eq = torch.where(mask, pos, torch.full_like(pos, -1)).max(dim=1).values
    has_eq = last_eq >= 0
    if has_eq.sum() == 0:
        return all_acc.item(), float("nan")
    rows = torch.arange(targets.shape[0], device=targets.device)[has_eq]
    last_pred = preds[rows, last_eq[has_eq]]
    last_tgt = targets[rows, last_eq[has_eq]]
    last_acc = (last_pred == last_tgt).float().mean()
    return all_acc.item(), last_acc.item()


def _train_and_eval(model, tag: str, cfg: SatCountConfig, num_classes: int,
                    device: str, chunk_divisor: int = 1):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    desc = f"satcount [{tag:<14s}]"
    bar = tqdm(range(cfg.steps), desc=desc, mininterval=0.5)

    model.train()
    for step in bar:
        input_ids, target_ids = make_satcount_batch(
            cfg.batch_size, cfg.L_train, cfg.K, cfg.eq_prob, device
        )
        logits = model(input_ids)                                # (B, L, V_out)
        # CE only over the K+1 valid output classes; ignore non-'=' positions.
        ce_logits = logits[..., :num_classes]
        loss = F.cross_entropy(
            ce_logits.reshape(-1, num_classes),
            target_ids.reshape(-1),
            ignore_index=IGNORE,
        )
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            all_acc, _ = _eq_acc(logits, target_ids, num_classes)
        bar.set_postfix_str(
            f"loss={loss.item():.3f} acc(=)={all_acc:.2%} grad_norm={grad_norm:.3f}"
        )

    model.eval()
    eval_B = max(cfg.batch_size, 128)
    all_acc, last_acc = {}, {}
    with torch.no_grad():
        for L in cfg.L_eval:
            assert L % chunk_divisor == 0, (
                f"L={L} not divisible by chunk_divisor={chunk_divisor}"
            )
            input_ids, target_ids = make_satcount_batch(
                eval_B, L, cfg.K, cfg.eq_prob, device
            )
            logits = model(input_ids)
            a, l = _eq_acc(logits, target_ids, num_classes)
            all_acc[L] = a
            last_acc[L] = l
    return {"all_acc": all_acc, "last_acc": last_acc}


# ---------------------------------------------------------------------------
# variant runners
# ---------------------------------------------------------------------------


def run_variant(boundary_nonlin: str, cfg: SatCountConfig, num_classes: int,
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


def run_ref_m2rnn(cfg: SatCountConfig, num_classes: int, vocab_size: int,
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
    cfg = SatCountConfig()
    device = _pick_device()

    num_classes = cfg.K + 1
    # vocab_size is shared by embed (3 input tokens) and output (K+1 classes).
    vocab_size = max(INPUT_VOCAB, num_classes)

    print(f"device={device}")
    print(
        f"saturated-counter | K={cfg.K}  eq_prob={cfg.eq_prob}  "
        f"input_vocab={INPUT_VOCAB}  out_classes={num_classes}  "
        f"train L={cfg.L_train}  eval L={list(cfg.L_eval)}"
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
        print(f"\n=== saturated-counter {title} ===")
        header = f"  {'variant':<14s} " + "".join(f"{f'L={L}':>10s} " for L in cfg.L_eval)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for kind, r in results.items():
            row = (
                f"  {kind:<14s} "
                + "".join(f"{r[key][L]:>10.2%} " for L in cfg.L_eval)
            )
            print(row)

    _print_table("all-'=' accuracy", "all_acc")
    _print_table("last-'=' accuracy", "last_acc")


if __name__ == "__main__":
    main()
