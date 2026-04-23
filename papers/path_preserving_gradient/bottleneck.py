"""
Reproduce the gradient-bottleneck phenomenon from
Godey & Artzi 2026 ("Lost in Backpropagation: The LM Head is a Gradient
Bottleneck", arXiv:2603.10145) on a tiny softmax policy, and check whether
logit-level natural gradient (NG-logit / Path) helps.

Setup:
    - Vocab size V = 50
    - Hidden dim D varied in {2, 5, 20, 50}
    - Policy is exactly a "residual stream + LM head":
          logits = W @ h,   h in R^D, W in R^{V x D}
      so D < V means the LM head is rank-deficient.
    - Target B is a categorical with mass on 3 tokens (easily expressible
      even at D = 2 since rank(log B_clipped) is small).
    - Compare pre-training of this policy under:
        * MLE  (vanilla cross-entropy grad)
        * NG-logit  (undamped 1/p - mean preconditioning at the logits)
        * NG-full   (full-parameter natural gradient with TRPO step)

Two artifacts are saved into this folder:
    - bottleneck_grad_destroyed.png : diagnostic showing how much of the
      logit-gradient norm is projected into ker(W^T) after NG-logit vs
      plain MLE. Illustrates the paper's main measurement.
    - bottleneck_training.png : loss curves per D per method.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

FIG_DIR = os.path.dirname(os.path.abspath(__file__))


def save_fig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved figure to {path}")


# ============================================================
# Config
# ============================================================
V = 50
D_VALUES = [2, 5, 20, 50]
N_SAMPLES = 128
N_STEPS = 300
N_UPDATES = 4
LR = 1e-2


def make_target(V, seed=1):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(V, generator=g)[:3]
    b = torch.full((V,), 1e-4)
    b[idx[0]] = 0.80
    b[idx[1]] = 0.15
    b[idx[2]] = 0.05 - 1e-4 * (V - 3)
    return b / b.sum()


B = make_target(V)


def make_policy(D, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = nn.Parameter(torch.randn(D, generator=g) * 0.5)
    W = nn.Parameter(torch.randn(V, D, generator=g) * (1.0 / D ** 0.5))
    return nn.ParameterDict({"h": h, "W": W})


def forward(policy):
    return policy["W"] @ policy["h"]


def kl(p, q):
    return torch.sum(p * (torch.log(p + 1e-12) - torch.log(q + 1e-12)))


# ============================================================
# NG-logit (same as path_preserving_gradient/pg.py, undamped)
# ============================================================
class NPGLogitAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta):
        ctx.save_for_backward(theta)
        return theta.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (theta,) = ctx.saved_tensors
        probs = F.softmax(theta, dim=-1)
        modified = grad_output / probs.clamp(min=1e-12)
        modified = modified - modified.mean(dim=-1, keepdim=True)
        return modified


# ============================================================
# Gradient bottleneck diagnostic
# ============================================================
def fraction_grad_destroyed(policy, transform):
    """Fraction of logit-gradient norm projected into ker(W^T).

    transform is either None (MLE) or a function applied to the logit
    gradient in V-space before the W projection.
    """
    W = policy["W"].detach()  # [V, D]
    logits = forward(policy)
    log_probs = F.log_softmax(logits, dim=-1)
    actions = torch.distributions.Categorical(probs=B).sample((1024,))
    loss = -log_probs[actions].mean()

    g_L = torch.autograd.grad(loss, logits, create_graph=False)[0]
    if transform is not None:
        g_L = transform(g_L, logits.detach())

    U, _ = torch.linalg.qr(W)
    g_row = U @ (U.T @ g_L)
    g_ker = g_L - g_row
    return (g_ker.norm() / g_L.norm().clamp(min=1e-12)).item()


def ng_logit_transform(g, theta):
    probs = F.softmax(theta, dim=-1)
    modified = g / probs.clamp(min=1e-12)
    modified = modified - modified.mean(dim=-1, keepdim=True)
    return modified


# ============================================================
# Training step functions
# ============================================================
def step_mle(policy, optimizer, n_updates=1):
    with torch.no_grad():
        actions = torch.distributions.Categorical(probs=B).sample((N_SAMPLES,))

    for _ in range(n_updates):
        logits = forward(policy)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -log_probs[actions].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def step_ng_logit(policy, optimizer, n_updates=1):
    with torch.no_grad():
        actions = torch.distributions.Categorical(probs=B).sample((N_SAMPLES,))

    for _ in range(n_updates):
        raw_logits = forward(policy)
        logits = NPGLogitAutograd.apply(raw_logits)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -log_probs[actions].mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1e2)
        optimizer.step()

    return loss.item()


def step_ng_full(policy, optimizer, n_updates=1, ridge=1e-3, kl_budget=1e-2):
    params = [policy["h"], policy["W"]]
    n_params = sum(p.numel() for p in params)

    for _ in range(n_updates):
        with torch.no_grad():
            actions = torch.distributions.Categorical(probs=B).sample((N_SAMPLES,))

        logits = forward(policy)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            probs = logits.softmax(dim=-1)

        scores = []
        for a in range(V):
            retain = a < V - 1
            grads_a = torch.autograd.grad(
                log_probs[a], params, retain_graph=retain
            )
            scores.append(torch.cat([g.reshape(-1) for g in grads_a]))
        scores = torch.stack(scores)

        g_flat = -scores[actions].mean(dim=0)

        Fmat = scores.t() @ (probs.unsqueeze(1) * scores)
        Fmat_damped = Fmat + ridge * torch.eye(
            n_params, device=Fmat.device, dtype=Fmat.dtype
        )

        try:
            natgrad = torch.linalg.solve(Fmat_damped, g_flat)
        except RuntimeError:
            natgrad = g_flat.clone()

        if not torch.isfinite(natgrad).all():
            natgrad = g_flat.clone()

        xFx = (natgrad @ (Fmat @ natgrad)).clamp(min=1e-12)
        alpha = torch.sqrt(2.0 * kl_budget / xFx).clamp(max=1.0)

        # Clip parameter step norm as an extra safety net
        step_vec = alpha * natgrad
        step_norm = step_vec.norm().clamp(min=1e-12)
        max_step = 10.0
        if step_norm > max_step:
            step_vec = step_vec * (max_step / step_norm)

        with torch.no_grad():
            offset = 0
            for p in params:
                n = p.numel()
                p.add_(-step_vec[offset : offset + n].view_as(p))
                offset += n

        optimizer.zero_grad()

    return float(-log_probs[actions].mean().item())


# ============================================================
# Training loop
# ============================================================
def train(D, step_fn, seed=0):
    torch.manual_seed(seed)
    policy = make_policy(D, seed=seed)
    optimizer = torch.optim.SGD(policy.parameters(), lr=LR)

    kl_hist = []
    for step in range(N_STEPS):
        step_fn(policy, optimizer, N_UPDATES)
        with torch.no_grad():
            probs = F.softmax(forward(policy), dim=-1)
            kl_hist.append(float(kl(B, probs).item()))
    return policy, kl_hist


# ============================================================
# Experiments
# ============================================================
def diagnostic_grad_destroyed():
    rows = []
    for D in D_VALUES:
        policy = make_policy(D, seed=0)
        frac_mle = fraction_grad_destroyed(policy, transform=None)
        frac_ng = fraction_grad_destroyed(policy, transform=ng_logit_transform)
        rows.append((D, frac_mle, frac_ng))
        print(
            f"D={D:3d} | frac destroyed (MLE) = {frac_mle:.3f} | "
            f"frac destroyed (NG-logit) = {frac_ng:.3f}"
        )

    fig, ax = plt.subplots(figsize=(5, 3.2))
    Ds = [r[0] for r in rows]
    mle = [r[1] for r in rows]
    ng = [r[2] for r in rows]
    x = np.arange(len(Ds))
    w = 0.38
    ax.bar(x - w / 2, mle, width=w, label="MLE grad")
    ax.bar(x + w / 2, ng, width=w, label="NG-logit grad")
    ax.set_xticks(x)
    ax.set_xticklabels([f"D={d}" for d in Ds])
    ax.set_ylabel("Fraction of grad norm in ker(W^T)")
    ax.set_title("LM-head gradient destroyed by backprop")
    ax.legend()
    plt.tight_layout()
    save_fig("bottleneck_grad_destroyed.png")
    plt.close(fig)


def training_curves():
    methods = {
        "MLE": step_mle,
        "NG-logit": step_ng_logit,
        "NG-full": step_ng_full,
    }

    fig, axes = plt.subplots(1, len(D_VALUES), figsize=(4 * len(D_VALUES), 3.2),
                             sharey=True)
    if len(D_VALUES) == 1:
        axes = [axes]

    for ax, D in zip(axes, D_VALUES):
        for name, fn in methods.items():
            _, kl_hist = train(D, fn, seed=0)
            ax.plot(kl_hist, label=name)
            print(
                f"D={D:3d} | {name:<9s} | final KL(B||pi) = {kl_hist[-1]:.4f}"
            )
        ax.set_title(f"D = {D}  (V = {V})")
        ax.set_xlabel("Outer step")
        ax.set_ylabel("KL(B || pi)")
        ax.set_yscale("log")
        ax.legend()
    plt.tight_layout()
    save_fig("bottleneck_training.png")
    plt.close(fig)


if __name__ == "__main__":
    print("=== Gradient-bottleneck diagnostic ===")
    diagnostic_grad_destroyed()
    print("\n=== Training curves ===")
    training_curves()
