"""
TTT-E2E Toy Implementation — Switching Bigram Sweep
====================================================
Based on Figure 2 (right panel) from:
  "End-to-End Test-Time Training for Long Context"
  (Tandon, Dalal, Li et al., arXiv:2512.23675)

Sweeps segment length M (how often the active bigram switches) over {4, 8, 16, 64}.
Four methods compared:
  1. Marginal MLP              — red line    — same arch, trained on NTP only, no memory
  2. No TTT                    — green line  — TTT arch without inner-loop adaptation
  3. Transformer with attention — orange line — full causal self-attention (ALiBi)
  4. TTT-E2E                   — blue line   — test-time training with meta-learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

# ══════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════
VOCAB       = 32
DIM         = 64
HIDDEN      = 4 * DIM
N_BLOCKS    = 2
N_HEADS     = 4
SEQ_LEN     = 64
INNER_LR    = 0.01
OUTER_LR    = 3e-4
TRAIN_STEPS = 500
N_EVAL      = 200
N_DIST      = 8
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
M_VALUES    = [4, 8, 16, 64]

torch.manual_seed(42)

# ══════════════════════════════════════════════════════════════
# Synthetic data: switching bigram distributions
# ══════════════════════════════════════════════════════════════
BIGRAMS = [
    torch.softmax(torch.randn(VOCAB, VOCAB, device=DEVICE) / 0.15, dim=-1)
    for _ in range(N_DIST)
]

CUR_M = None  # segment length, set per sweep iteration

def sample_seq(length):
    toks = [torch.randint(VOCAB, (1,), device=DEVICE).item()]
    while len(toks) < length:
        bg = BIGRAMS[torch.randint(N_DIST, (1,)).item()]
        seg_len = min(CUR_M, length - len(toks))
        for _ in range(seg_len):
            toks.append(torch.multinomial(bg[toks[-1]], 1).item())
    return torch.tensor(toks, device=DEVICE)


# ══════════════════════════════════════════════════════════════
# 1. ATTENTION BASELINE (standard Transformer, nn.Module)
# ══════════════════════════════════════════════════════════════

def _alibi_slopes(n_heads):
    ratio = 2 ** (-8.0 / n_heads)
    return torch.tensor([ratio ** i for i in range(1, n_heads + 1)])


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.register_buffer("alibi_slopes", _alibi_slopes(n_heads))

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        pos = torch.arange(T, device=x.device)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
        alibi = self.alibi_slopes.view(1, -1, 1, 1) * dist
        attn = attn - alibi

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.ln1  = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.ln2  = nn.LayerNorm(dim)
        self.mlp  = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(VOCAB, DIM)
        self.blocks = nn.Sequential(*[TransformerBlock(DIM, N_HEADS) for _ in range(N_BLOCKS)])
        self.ln   = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, tokens):
        h = self.emb(tokens)
        h = self.blocks(h)
        h = self.ln(h)
        return self.head(h)


# ══════════════════════════════════════════════════════════════
# 2. TTT-E2E MODEL (functional style, no nn.Module)
# ══════════════════════════════════════════════════════════════

def init_params():
    s = (2 / (5 * DIM)) ** 0.5
    p = {}
    p["emb"]   = torch.randn(VOCAB, DIM, device=DEVICE) * 0.02
    for i in range(N_BLOCKS):
        p[f"{i}.nw"] = torch.ones(DIM, device=DEVICE)
        p[f"{i}.nb"] = torch.zeros(DIM, device=DEVICE)
        p[f"{i}.w1"] = torch.randn(HIDDEN, DIM, device=DEVICE) * s
        p[f"{i}.b1"] = torch.zeros(HIDDEN, device=DEVICE)
        p[f"{i}.w2"] = torch.randn(DIM, HIDDEN, device=DEVICE) * s
        p[f"{i}.b2"] = torch.zeros(DIM, device=DEVICE)
    p["fnw"]  = torch.ones(DIM, device=DEVICE)
    p["fnb"]  = torch.zeros(DIM, device=DEVICE)
    p["head"] = torch.randn(VOCAB, DIM, device=DEVICE) * 0.02
    for v in p.values():
        v.requires_grad_(True)
    return p

_tmp = init_params()
TTT_KEYS = sorted([k for k in _tmp if "w1" in k or "b1" in k or "w2" in k or "b2" in k])
FROZEN_KEYS = sorted([k for k in _tmp if k not in TTT_KEYS])
del _tmp


def fwd(tok_idx, p):
    h = p["emb"][tok_idx].unsqueeze(0)
    for i in range(N_BLOCKS):
        r = h
        h = F.layer_norm(h, (DIM,), p[f"{i}.nw"], p[f"{i}.nb"])
        h = F.gelu(F.linear(h, p[f"{i}.w1"], p[f"{i}.b1"]))
        h = F.linear(h, p[f"{i}.w2"], p[f"{i}.b2"])
        h = r + h
    h = F.layer_norm(h, (DIM,), p["fnw"], p["fnb"])
    return F.linear(h, p["head"])


def ttt_e2e_train_loss(params, tokens):
    T = len(tokens) - 1
    w      = {k: params[k] for k in TTT_KEYS}
    frozen = {k: params[k] for k in FROZEN_KEYS}
    total_loss = torch.tensor(0.0, device=DEVICE)
    for t in range(T):
        cur = {**frozen, **w}
        logits = fwd(tokens[t], cur)
        ell_t  = F.cross_entropy(logits, tokens[t + 1].unsqueeze(0))
        total_loss = total_loss + ell_t
        grads = torch.autograd.grad(ell_t, list(w.values()), create_graph=True)
        w = {k: w[k] - INNER_LR * g for k, g in zip(TTT_KEYS, grads)}
    return total_loss / T


def eval_with_ttt(params, tokens):
    T = len(tokens) - 1
    w      = {k: params[k].detach().clone().requires_grad_(True) for k in TTT_KEYS}
    frozen = {k: params[k].detach() for k in FROZEN_KEYS}
    losses = []
    for t in range(T):
        cur = {**frozen, **w}
        logits = fwd(tokens[t], cur)
        ell_t  = F.cross_entropy(logits, tokens[t + 1].unsqueeze(0))
        losses.append(ell_t.item())
        grads = torch.autograd.grad(ell_t, list(w.values()))
        w = {k: (w[k] - INNER_LR * g).detach().requires_grad_(True)
             for k, g in zip(TTT_KEYS, grads)}
    return losses


def eval_without_ttt(params, tokens):
    T = len(tokens) - 1
    losses = []
    with torch.no_grad():
        for t in range(T):
            logits = fwd(tokens[t], params)
            losses.append(F.cross_entropy(logits, tokens[t+1].unsqueeze(0)).item())
    return losses


# ══════════════════════════════════════════════════════════════
# Switching bigram sweep: train + eval for each M
# ══════════════════════════════════════════════════════════════
print(f"Device: {DEVICE}")
print(f"Switching bigram sweep over M = {M_VALUES}\n")

all_results = {}

for M in M_VALUES:
    print(f"\n{'#'*60}")
    print(f"#  M = {M}  (segment length)")
    print(f"{'#'*60}")

    CUR_M = M

    # --- Train attention baseline ---
    print(f"\n  [M={M}] Phase 1: Training attention baseline ({TRAIN_STEPS} steps)")
    attn_model = TransformerLM().to(DEVICE)
    attn_opt   = torch.optim.Adam(attn_model.parameters(), lr=OUTER_LR)

    t0 = time.time()
    for step in range(TRAIN_STEPS):
        tokens = sample_seq(SEQ_LEN + 1).unsqueeze(0)
        logits = attn_model(tokens[:, :-1])
        loss = F.cross_entropy(logits.view(-1, VOCAB), tokens[:, 1:].reshape(-1))
        attn_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(attn_model.parameters(), 1.0)
        attn_opt.step()
        if (step + 1) % 100 == 0:
            print(f"    step {step+1:>4d}/{TRAIN_STEPS}  loss={loss.item():.4f}  ({time.time()-t0:.0f}s)")
    print(f"    Done in {time.time()-t0:.0f}s")

    # --- Train marginal MLP baseline ---
    print(f"\n  [M={M}] Phase 2: Training marginal MLP baseline ({TRAIN_STEPS} steps)")
    marg_params = init_params()
    marg_opt = torch.optim.Adam(marg_params.values(), lr=OUTER_LR)

    t0 = time.time()
    for step in range(TRAIN_STEPS):
        tokens = sample_seq(SEQ_LEN + 1)
        marg_opt.zero_grad()
        T = len(tokens) - 1
        loss = torch.tensor(0.0, device=DEVICE)
        for t in range(T):
            logits = fwd(tokens[t], marg_params)
            loss = loss + F.cross_entropy(logits, tokens[t + 1].unsqueeze(0))
        loss = loss / T
        loss.backward()
        torch.nn.utils.clip_grad_norm_(marg_params.values(), 1.0)
        marg_opt.step()
        if (step + 1) % 100 == 0:
            print(f"    step {step+1:>4d}/{TRAIN_STEPS}  loss={loss.item():.4f}  ({time.time()-t0:.0f}s)")
    print(f"    Done in {time.time()-t0:.0f}s")

    # --- Train TTT-E2E ---
    print(f"\n  [M={M}] Phase 3: Training TTT-E2E ({TRAIN_STEPS} steps)")
    params = init_params()
    ttt_opt = torch.optim.Adam(params.values(), lr=OUTER_LR)

    t0 = time.time()
    for step in range(TRAIN_STEPS):
        tokens = sample_seq(SEQ_LEN + 1)
        ttt_opt.zero_grad()
        loss = ttt_e2e_train_loss(params, tokens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params.values(), 1.0)
        ttt_opt.step()
        if (step + 1) % 100 == 0:
            print(f"    step {step+1:>4d}/{TRAIN_STEPS}  loss={loss.item():.4f}  ({time.time()-t0:.0f}s)")
    print(f"    Done in {time.time()-t0:.0f}s")

    # --- Evaluate ---
    print(f"\n  [M={M}] Evaluating all 4 methods on {N_EVAL} test sequences...")
    avg_ttt   = [0.0] * SEQ_LEN
    avg_nottt = [0.0] * SEQ_LEN
    avg_attn  = [0.0] * SEQ_LEN
    avg_marg  = [0.0] * SEQ_LEN

    attn_model.eval()
    for s in range(N_EVAL):
        seq = sample_seq(SEQ_LEN + 1)

        lt = eval_with_ttt(params, seq)
        ln = eval_without_ttt(params, seq)
        lm = eval_without_ttt(marg_params, seq)

        with torch.no_grad():
            logits = attn_model(seq[:-1].unsqueeze(0))
            for t in range(SEQ_LEN):
                la = F.cross_entropy(logits[0, t:t+1], seq[t+1:t+2]).item()
                avg_attn[t] += la / N_EVAL

        for i in range(SEQ_LEN):
            avg_ttt[i]   += lt[i] / N_EVAL
            avg_nottt[i] += ln[i] / N_EVAL
            avg_marg[i]  += lm[i] / N_EVAL

        if (s+1) % 50 == 0:
            print(f"    {s+1}/{N_EVAL} done")

    all_results[M] = {
        "marg": avg_marg,
        "nottt": avg_nottt,
        "attn": avg_attn,
        "ttt": avg_ttt,
    }

    # Print table for this M
    print(f"\n  [M={M}] Results:")
    print(f"  {'Token t':>8s}  {'Marginal':>8s}  {'No TTT':>8s}  {'Attn':>8s}  {'TTT-E2E':>9s}")
    print(f"  {'─' * 50}")
    for t in [0, 3, 7, 15, 31, 47, 63]:
        if t < SEQ_LEN:
            print(f"  {t+1:>8d}  {avg_marg[t]:>8.4f}  {avg_nottt[t]:>8.4f}  {avg_attn[t]:>8.4f}  {avg_ttt[t]:>9.4f}")
    m_ttt   = sum(avg_ttt)   / SEQ_LEN
    m_nottt = sum(avg_nottt) / SEQ_LEN
    m_attn  = sum(avg_attn)  / SEQ_LEN
    m_marg  = sum(avg_marg)  / SEQ_LEN
    print(f"  {'─' * 50}")
    print(f"  {'mean':>8s}  {m_marg:>8.4f}  {m_nottt:>8.4f}  {m_attn:>8.4f}  {m_ttt:>9.4f}")


# ══════════════════════════════════════════════════════════════
# 2x2 subplot figure
# ══════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
xs = range(1, SEQ_LEN + 1)

for ax, M in zip(axes.flat, M_VALUES):
    r = all_results[M]
    ax.plot(xs, r["marg"],  label="Marginal MLP (no memory)",        color="#d62728", lw=2.2)
    ax.plot(xs, r["nottt"], label="No TTT",                          color="#2ca02c", lw=2.2)
    ax.plot(xs, r["attn"],  label="Transformer with attention",      color="#ff7f0e", lw=2.2)
    ax.plot(xs, r["ttt"],   label="TTT-E2E",                         color="#1f77b4", lw=2.2)
    ax.set_title(f"M = {M}  (segment length)", fontsize=13)
    ax.grid(True, alpha=0.3)

axes[1, 0].set_xlabel("Token index t", fontsize=12)
axes[1, 1].set_xlabel("Token index t", fontsize=12)
axes[0, 0].set_ylabel("Loss (log perplexity)", fontsize=12)
axes[1, 0].set_ylabel("Loss (log perplexity)", fontsize=12)

axes[0, 0].legend(fontsize=9)
fig.suptitle("TTT-E2E: Switching Bigram Sweep", fontsize=15, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("ttt_e2e_switching_bigram_sweep.png", dpi=150)
print("\nPlot saved to ttt_e2e_switching_bigram_sweep.png!")
