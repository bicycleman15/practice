"""
Multi-Query Associative Recall (MQAR) task for NonLinearLinearRNNLM.

Sequence layout (length = 2*(N + M)):

    k_1 v_1  k_2 v_2  ...  k_N v_N     q_1 a_1  q_2 a_2  ...  q_M a_M
    -----  memorization phase  -----   -------   recall phase   -------

- k_i are sampled without replacement from the vocab (unique keys per sample).
- v_i are sampled uniformly from the vocab (values need not be unique).
- q_j is a random key already seen; a_j is its paired value.
- Loss is computed only at the `a_j` positions (the tokens the model must recall).

This is the benchmark that separates "state-tracking" linear RNNs from vanilla
linear attention in Arora et al. 2024 (Zoology) and NVLabs GatedDeltaNet.

Run with:a
    source ~/.zshrc && conda activate t3
    python -m rnn.non_linear_linear_rnn_mqar
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from tqdm import tqdm

from rnn.non_linear_linear_rnn import NonLinearLinearRNNConfig, NonLinearLinearRNNLM


# ---------------------------------------------------------------------------
# MQAR data
# ---------------------------------------------------------------------------


def make_mqar_batch(batch_size: int, num_pairs: int, num_queries: int, vocab_size: int, device):
    """Returns (input_ids, target_ids, loss_mask).

    Shapes:
        input_ids, target_ids: (B, L)
        loss_mask: (B, L) bool, True only at the answer positions (where we
                   want the model to predict v).
    L = 2*(num_pairs + num_queries).
    """
    B, N, M, V = batch_size, num_pairs, num_queries, vocab_size
    assert N <= V, f"need vocab_size >= num_pairs to sample unique keys ({V} < {N})"

    # Unique keys per sample (sample without replacement via argsort trick).
    keys = torch.argsort(torch.rand(B, V, device=device), dim=-1)[:, :N]          # (B, N)
    values = torch.randint(0, V, (B, N), device=device)                             # (B, N)

    # Memorization phase: interleave keys and values.
    mem = torch.stack([keys, values], dim=-1).reshape(B, 2 * N)                     # (B, 2N)

    # Recall phase: pick M queries = random indices into the N pairs.
    q_idx = torch.randint(0, N, (B, M), device=device)                              # (B, M)
    q_keys = torch.gather(keys, 1, q_idx)                                            # (B, M)
    q_answers = torch.gather(values, 1, q_idx)                                       # (B, M)
    recall = torch.stack([q_keys, q_answers], dim=-1).reshape(B, 2 * M)              # (B, 2M)

    # Pad at the end so input_ids (after the next-token shift) has length 2*(N+M),
    # which is what we need to be divisible by chunk_size.
    pad = torch.zeros(B, 1, dtype=torch.long, device=device)
    tokens = torch.cat([mem, recall, pad], dim=1)                                    # (B, 2(N+M) + 1)
    input_ids = tokens[:, :-1]                                                        # (B, 2(N+M))
    target_ids = tokens[:, 1:]                                                         # (B, 2(N+M))

    # Loss mask: True at positions i where target_ids[i] is an answer token.
    # Answer a_j is at position 2N + 2j + 1 in `tokens`, so at index 2N + 2j in target_ids.
    L = input_ids.shape[1]
    loss_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    for j in range(M):
        loss_mask[:, 2 * N + 2 * j] = True

    return input_ids, target_ids, loss_mask


# ---------------------------------------------------------------------------
# training config
# ---------------------------------------------------------------------------


@dataclass
class MQARConfig:
    # task  (2*(N+M) must be divisible by chunk_size after the trailing pad)
    num_pairs: int = 24
    num_queries: int = 8
    vocab_size: int = 64
    batch_size: int = 32

    # optim
    steps: int = 500
    lr: float = 3e-3
    seed: int = 0
    log_every: int = 25

    # model
    dim: int = 96
    num_heads: int = 4
    head_k_dim: int = 24
    head_v_dim: int = 24
    layers: int = 2
    chunk_size: int = 16        # seq_len = 2*(24+8) = 64  =>  4 chunks
    use_short_conv: bool = True

    # which boundary non-linearities to sweep
    variants: tuple = ("identity", "rmsnorm", "tanh_res", "gru")


def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# train/eval for one boundary variant
# ---------------------------------------------------------------------------


def run_variant(boundary_nonlin: str, decay_init: str, cfg: MQARConfig, device: str):
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
        decay_init=decay_init,
    )
    model = NonLinearLinearRNNLM(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    seq_len = 2 * (cfg.num_pairs + cfg.num_queries) - 1  # after the shift for next-token prediction
    desc = f"mqar [{boundary_nonlin:<9s} decay={decay_init:<6s}]"
    bar = tqdm(range(cfg.steps), desc=desc, mininterval=0.5, miniters=cfg.log_every)

    last_loss = float("nan")
    last_acc = 0.0
    best_acc = 0.0
    for step in bar:
        input_ids, target_ids, loss_mask = make_mqar_batch(
            cfg.batch_size, cfg.num_pairs, cfg.num_queries, cfg.vocab_size, device,
        )
        logits = model(input_ids)                                  # (B, L, V)
        logits_flat = logits.reshape(-1, cfg.vocab_size)
        targets_flat = target_ids.reshape(-1)
        mask_flat = loss_mask.reshape(-1)

        # cross entropy only on recall positions
        ce = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat])

        optimizer.zero_grad()
        ce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = (preds == target_ids) & loss_mask
            acc = correct.sum().float().item() / loss_mask.sum().float().item()

        last_loss = ce.item()
        last_acc = acc
        best_acc = max(best_acc, acc)
        if (step + 1) % cfg.log_every == 0 or step == cfg.steps - 1:
            bar.set_postfix_str(
                f"loss={last_loss:.3f} acc={acc:.2%} best={best_acc:.2%}"
            )

    return {
        "variant": boundary_nonlin,
        "decay_init": decay_init,
        "final_loss": last_loss,
        "final_acc": last_acc,
        "best_acc": best_acc,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    cfg = MQARConfig()
    device = _pick_device()
    L_total = 2 * (cfg.num_pairs + cfg.num_queries)
    print(
        f"device={device}  "
        f"N_pairs={cfg.num_pairs}  N_queries={cfg.num_queries}  "
        f"seq_len={L_total}  vocab={cfg.vocab_size}  "
        f"chunk_size={cfg.chunk_size}  (=> {L_total // cfg.chunk_size} chunks)"
    )
    print(f"random-guess accuracy: {1.0 / cfg.vocab_size:.2%}\n")

    # Focused sweep: for each boundary f, try both mild and strong intra-chunk decay.
    # The hackmd proposal argues that within-chunk decay + boundary nonlinearity
    # together give a "fast forgetting / slow consolidation" split.
    configurations = [(kind, di) for kind in cfg.variants for di in ("mild", "strong")]

    results = []
    for kind, di in configurations:
        results.append(run_variant(kind, di, cfg, device))

    print("\n=== MQAR summary ===")
    header = (
        f"  {'variant':<10s} {'decay':<8s} "
        f"{'final_loss':>12s} {'final_acc':>12s} {'best_acc':>12s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        print(
            f"  {r['variant']:<10s} {r['decay_init']:<8s} "
            f"{r['final_loss']:>12.4f} "
            f"{r['final_acc']:>12.2%} "
            f"{r['best_acc']:>12.2%}"
        )


if __name__ == "__main__":
    main()
