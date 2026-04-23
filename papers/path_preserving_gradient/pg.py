import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

torch.manual_seed(0)

FIG_DIR = os.path.dirname(os.path.abspath(__file__))

def save_fig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved figure to {path}")

def set_seed(seed=0):
    torch.manual_seed(seed)

# ============================================================
# Config
# ============================================================
n_inputs = 4
hidden_size = 16
n_actions = 5

X = torch.tensor([[0.2, -0.5, 0.3, 0.1]])  # fixed input to network
Rewards = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])  # reward function
B = torch.tensor([0.98, 0.0, 0.01, 0.01, 0.1])

# ============================================================
# Policy
# ============================================================
def make_policy():
    return nn.Sequential(
        nn.Linear(n_inputs, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, n_actions),
    )

def forward(policy):
    logits = policy(X)[0]
    probs = F.softmax(logits, dim=-1)
    return logits, probs


def expected_reward(probs):
    return torch.sum(probs * Rewards)


def compute_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-8))


def kl_divergence(p, q):
    return torch.sum(p * torch.log((p + 1e-8) / (q + 1e-8)))


def warm_start(policy, steps=20, lr=0.1):
    # Move the initial distribution to a difficult place
    optimizer = torch.optim.SGD(policy.parameters(), lr=lr)

    # hard-coded target distribution
    target = torch.tensor([0, 0.7, 0.1, 0.1, 0.1])

    for _ in range(steps):
        logits = policy(X)[0]
        probs = torch.softmax(logits, dim=-1)

        # minimize KL(target || policy)
        loss = -(target * torch.log(probs + 1e-8)).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def plot_example_warm_start():
    import matplotlib.pyplot as plt
    import numpy as np

    policy = make_policy()
    warm_start(policy, steps=500)

    logits, probs = forward(policy)
    policy_reward = expected_reward(probs)

    # convert tensors if needed
    target = B.detach().numpy() if torch.is_tensor(B) else np.array(B)
    target_reward = expected_reward(torch.tensor(target))
    reward = Rewards.detach().numpy() if torch.is_tensor(Rewards) else np.array(Rewards)

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))

    # left: reward
    axes[0].bar(np.arange(5), reward, color="darkorange")
    axes[0].set_xticks(np.arange(5))
    axes[0].set_xlabel("Action")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Reward Function")

    # middle: learned policy
    axes[1].bar(np.arange(5), probs.detach().numpy(), color="steelblue")
    axes[1].set_xticks(np.arange(5))
    axes[1].set_xlabel("Action")
    axes[1].set_ylabel("Probability")
    axes[1].set_title(f"Policy Probs (E[R]={policy_reward:.2f})")

    # right: target distribution B
    axes[2].bar(np.arange(5), target, color="purple")
    axes[2].set_xticks(np.arange(5))
    axes[2].set_xlabel("Action")
    axes[2].set_title(f"Target B (E[R] = {target_reward:.2f})")

    plt.tight_layout()
    save_fig("warm_start.png")
    plt.close(fig)

plot_example_warm_start()

# ============================================================
# Training loop (simple + wrapped)
# ============================================================
def train(step_fn, n_steps=30, lr=0.1, alpha=0.5, print_every=10, n_updates=1, seed=0):
    set_seed(seed)
    policy = make_policy()
    # warm_start(policy, steps=500)

    optimizer = torch.optim.SGD(policy.parameters(), lr=lr)
    #optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    history = {
        "logits": [],
        "avg_logits": [],
        "probs": [],
        "expected_reward": [],
        "kl_pi_b": [],
        "kl_b_pi": [],
    }

    for step in range(n_steps):
        # --- update ---
        info = step_fn(policy, optimizer, n_updates)

        # --- record AFTER update ---
        with torch.no_grad():
            logits, probs = forward(policy)
            er = expected_reward(probs)

            kl_pi_b = kl_divergence(probs, B)
            kl_b_pi = kl_divergence(B, probs)

        history["logits"].append(logits.clone())
        history["avg_logits"].append(logits.mean().item())
        history["probs"].append(probs.clone())
        history["expected_reward"].append(er.item())
        history["kl_pi_b"].append(kl_pi_b.item())
        history["kl_b_pi"].append(kl_b_pi.item())

        if step % print_every == 0 or step == n_steps - 1:
            print(
                f"step={step:02d} | "
                f"mode={step_fn.__name__:<16} | "
                f"E[r]={er.item():.4f} | "
                f"avg_logit={logits.mean().item():.4f} | "
                f"kl_b_pi={kl_pi_b.item():.4f}"
            )

    return policy, history


def plot_history(history, b=None, save_name=None):
    """
    Expects history to contain:
      - history["expected_reward"] : list of floats
      - history["probs"]           : list of [n_actions] tensors
      - history["logits"]          : list of [n_actions] tensors

    Optional:
      - b : target distribution tensor of shape [n_actions]
    """
    probs_hist = [p.detach().cpu() for p in history["probs"]]
    logits_hist = [z.detach().cpu() for z in history["logits"]]
    rewards_hist = history["expected_reward"]
    steps = list(range(len(rewards_hist)))

    def entropy(p):
        return float(-(p * torch.log(p + 1e-8)).sum().item())

    def fkl(p, q):
        # forward KL: KL(p || q)
        return float((p * torch.log((p + 1e-8) / (q + 1e-8))).sum().item())

    def dkl(p, q):
        # reverse KL: KL(q || p)
        return float((q * torch.log((q + 1e-8) / (p + 1e-8))).sum().item())

    entropies = [entropy(p) for p in probs_hist]

    if b is not None:
        b = b.detach().cpu()
        fkl_to_b = [fkl(b, p) for p in probs_hist]   # KL(b || policy)
        dkl_to_b = [dkl(b, p) for p in probs_hist]   # KL(policy || b)
        diff_to_b = [(b - p).numpy() for p in probs_hist]
    else:
        fkl_to_b = None
        dkl_to_b = None
        diff_to_b = None

    n_actions = logits_hist[0].shape[0]

    plt.figure(figsize=(15, 3))

    # 1) expected reward
    plt.subplot(1, 5, 1)
    plt.plot(steps, rewards_hist)
    plt.xlabel("Step")
    plt.title("Expected Reward")

    # 2) distance to b
    plt.subplot(1, 5, 2)
    if b is not None:
        plt.plot(steps, fkl_to_b, label="KL(b || policy)")
        plt.plot(steps, dkl_to_b, label="KL(policy || b)")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No target b provided", ha="center", va="center")
    plt.xlabel("Step")
    plt.title("Distance to b")

    # 3) entropy
    plt.subplot(1, 5, 3)
    plt.plot(steps, entropies)
    plt.xlabel("Step")
    plt.title("Policy Entropy")

    # 4) per-logit evolution
    plt.subplot(1, 5, 4)
    for a in range(n_actions):
        plt.plot(steps, [z[a].item() for z in logits_hist], label=f"logit {a}")
    plt.xlabel("Step")
    plt.title("Logit Evolution")
    plt.legend()

    # 5) b - policy probabilities
    plt.subplot(1, 5, 5)
    if b is not None:
        for a in range(n_actions):
            plt.plot(steps, [d[a] for d in diff_to_b], label=f"a{a}")
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No target b provided", ha="center", va="center")
    plt.xlabel("Step")
    plt.title("b - policy probs")

    plt.tight_layout()
    if save_name is not None:
        save_fig(save_name)
        plt.close()
    else:
        plt.show()

N_SAMPLES = 128

def step_reinforce(policy, optimizer, n_updates=1):
    # Sample a fixed batch of actions from the current policy
    with torch.no_grad():
        logits, probs = forward(policy)
        pi_dist = torch.distributions.Categorical(logits=logits)
        actions = pi_dist.sample((N_SAMPLES,))  # shape: [n_samples]
        rewards = torch.as_tensor(
            [Rewards[int(a)] for a in actions],
            dtype=logits.dtype,
            device=logits.device,
        )

    # Centered rewards = baseline subtraction
    advantages = rewards - rewards.mean()

    for _ in range(n_updates):
        # Recompute policy after each optimizer step
        logits, probs = forward(policy)
        pi_dist = torch.distributions.Categorical(logits=logits)

        log_probs = pi_dist.log_prob(actions)   # shape: [n_samples]
        obj = (advantages * log_probs).mean()

        optimizer.zero_grad()
        (-obj).backward()
        optimizer.step()

    return {
        "mean_reward": rewards.mean().item(),
        "objective": obj.item(),
        "sampled_action": [int(a) for a in actions],
    }




def step_reinforce_pretrain(policy, optimizer, n_updates=1):
    # Pre-training REINFORCE: sample target from B, sample action from pi,
    # reward is 1 if they match, else 0.
    with torch.no_grad():
        logits, _ = forward(policy)
        pi_dist = torch.distributions.Categorical(logits=logits)
        actions = pi_dist.sample((N_SAMPLES,))

        target_dist = torch.distributions.Categorical(probs=B)
        a_target = target_dist.sample((N_SAMPLES,))

        rewards = (actions == a_target).to(logits.dtype)

    advantages = rewards - rewards.mean()

    for _ in range(n_updates):
        logits, _ = forward(policy)
        pi_dist = torch.distributions.Categorical(logits=logits)
        log_probs = pi_dist.log_prob(actions)
        obj = (advantages * log_probs).mean()

        optimizer.zero_grad()
        (-obj).backward()
        optimizer.step()

    return {
        "objective": obj.item(),
        "mean_reward": rewards.mean().item(),
        "sampled_action": [int(a) for a in actions],
    }


def step_fkl_b(policy, optimizer, n_updates=1):
    # Target distribution sampling
    with torch.no_grad():
        target_probs = B
        dist = torch.distributions.Categorical(probs=target_probs)
        actions = dist.sample((N_SAMPLES,))

    for _ in range(n_updates):
        logits, probs = forward(policy)
        cat = torch.distributions.Categorical(probs=probs)
        obj = cat.log_prob(actions).mean()

        optimizer.zero_grad()
        (-obj).backward()
        optimizer.step()

    return {
        "objective": obj.item(),
        "sampled_action": [int(a) for a in actions],
    }




class PathPreservingAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta):
        ctx.save_for_backward(theta)
        return theta.clone()

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        grad_input = grad_output.clone()

        probs = F.softmax(theta, dim=-1)

        modified_grad = grad_input / (probs + 1e-2)
        modified_grad = modified_grad - modified_grad.mean(dim=-1, keepdim=True)

        return modified_grad


class NPGLogitAutograd(torch.autograd.Function):
    """Undamped logit-level NPG: (F^+ g)_k = g_k / p_k - mean(g/p).

    Uses a tiny clamp on probs only to avoid division by zero; there is
    no eps-damping of the preconditioner magnitude itself.
    """
    @staticmethod
    def forward(ctx, theta):
        ctx.save_for_backward(theta)
        return theta.clone()

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        probs = F.softmax(theta, dim=-1)
        modified = grad_output / probs.clamp(min=1e-12)
        modified = modified - modified.mean(dim=-1, keepdim=True)
        return modified


def step_npg_logit(policy, optimizer, n_updates=1):
    # Same loss as step_b_path / step_fkl_b but with exact (undamped) NPG
    # preconditioning on the logits.
    with torch.no_grad():
        target_dist = torch.distributions.Categorical(probs=B)
        buf_actions = target_dist.sample((N_SAMPLES,))

    for _ in range(n_updates):
        raw_logits, _ = forward(policy)
        logits = NPGLogitAutograd.apply(raw_logits)
        cat = torch.distributions.Categorical(logits=logits)
        obj = cat.log_prob(buf_actions).mean()

        optimizer.zero_grad()
        (-obj).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1e2)
        optimizer.step()

    return {
        "objective": obj.item(),
        "sampled_actions": buf_actions.tolist(),
    }


def step_npg_full(policy, optimizer, n_updates=1, ridge=1e-6, kl_budget=1e-4):
    """Full-parameter NPG with a TRPO-style trust-region step size.

    Given natgrad x = F^{-1} g, takes step size alpha = sqrt(2 * kl_budget /
    (x^T F x)) so that one update changes KL(pi_new || pi_old) ~ kl_budget
    per inner update, decoupled from the optimizer's lr.
    """
    params = list(policy.parameters())
    n_params = sum(p.numel() for p in params)

    for _ in range(n_updates):
        with torch.no_grad():
            target_dist = torch.distributions.Categorical(probs=B)
            actions = target_dist.sample((N_SAMPLES,))

        logits, _ = forward(policy)
        log_probs_all = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            probs = logits.softmax(dim=-1)

        # Per-action score vectors s_a = grad log pi(a) over ALL params
        scores = []
        for a in range(n_actions):
            retain = a < n_actions - 1
            grads_a = torch.autograd.grad(
                log_probs_all[a], params, retain_graph=retain
            )
            scores.append(torch.cat([g.reshape(-1) for g in grads_a]))
        scores = torch.stack(scores)  # [n_actions, n_params]

        # MLE gradient as a linear combination of scores
        g_flat = -scores[actions].mean(dim=0)

        # Analytical Fisher over parameters
        Fmat = scores.t() @ (probs.unsqueeze(1) * scores)
        Fmat_damped = Fmat + ridge * torch.eye(
            n_params, device=Fmat.device, dtype=Fmat.dtype
        )

        natgrad = torch.linalg.solve(Fmat_damped, g_flat)

        # Trust-region step size: x^T F x ~ 2 * KL(pi_new || pi_old)
        xFx = (natgrad @ (Fmat @ natgrad)).clamp(min=1e-12)
        alpha = torch.sqrt(2.0 * kl_budget / xFx).clamp(max=1.0)

        with torch.no_grad():
            offset = 0
            for p in params:
                n = p.numel()
                p.add_(-alpha * natgrad[offset:offset + n].view_as(p))
                offset += n

        optimizer.zero_grad()

    loss_val = float(-log_probs_all[actions].mean().item())
    return {"objective": loss_val, "alpha": float(alpha.item())}


def step_b_path(policy, optimizer, n_updates=1):
    # Sample actions from target distribution B
    with torch.no_grad():
        target_probs = B
        dist = torch.distributions.Categorical(probs=target_probs)
        buf_actions = dist.sample((N_SAMPLES,))

    for _ in range(n_updates):
        raw_logits, _probs = forward(policy)

        logits = PathPreservingAutograd.apply(raw_logits)
        logits.retain_grad()

        cat = torch.distributions.Categorical(logits=logits)

        obj = cat.log_prob(buf_actions).mean()

        optimizer.zero_grad()
        (-obj).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1e2)
        optimizer.step()

    return {
        "objective": obj.item(),
        "sampled_actions": buf_actions.tolist(),
    }


def plot_comparison(histories, b, save_name):
    """
    histories: dict[name] -> history dict (as returned by train()).
    b: target distribution tensor of shape [n_actions].
    Produces a single figure with 4 panels comparing all methods.
    """
    import numpy as np

    b_cpu = b.detach().cpu()

    def entropy(p):
        return float(-(p * torch.log(p + 1e-8)).sum().item())

    def fkl(p, q):
        return float((p * torch.log((p + 1e-8) / (q + 1e-8))).sum().item())

    def dkl(p, q):
        return float((q * torch.log((q + 1e-8) / (p + 1e-8))).sum().item())

    fig, axes = plt.subplots(1, 4, figsize=(18, 3.5))

    for name, history in histories.items():
        probs_hist = [p.detach().cpu() for p in history["probs"]]
        steps = list(range(len(probs_hist)))

        fkl_to_b = [fkl(b_cpu, p) for p in probs_hist]
        dkl_to_b = [dkl(b_cpu, p) for p in probs_hist]
        entropies = [entropy(p) for p in probs_hist]

        axes[0].plot(steps, fkl_to_b, label=name)
        axes[1].plot(steps, dkl_to_b, label=name)
        axes[2].plot(steps, entropies, label=name)

    axes[0].set_title("KL(B || policy)")
    axes[0].set_xlabel("Step")
    axes[0].legend()

    axes[1].set_title("KL(policy || B)")
    axes[1].set_xlabel("Step")
    axes[1].legend()

    axes[2].set_title("Policy Entropy")
    axes[2].set_xlabel("Step")
    axes[2].legend()

    # Final-distribution bar chart: B plus one bar per method per action.
    n_actions = b_cpu.shape[0]
    n_series = 1 + len(histories)
    bar_w = 0.8 / n_series
    x = np.arange(n_actions)

    axes[3].bar(x - 0.4 + bar_w * 0.5, b_cpu.numpy(), width=bar_w, label="B", color="black")
    for i, (name, history) in enumerate(histories.items(), start=1):
        final_probs = history["probs"][-1].detach().cpu().numpy()
        axes[3].bar(x - 0.4 + bar_w * (i + 0.5), final_probs, width=bar_w, label=name)

    axes[3].set_xticks(x)
    axes[3].set_xlabel("Action")
    axes[3].set_ylabel("Probability")
    axes[3].set_title("Final Policy vs B")
    axes[3].legend()

    plt.tight_layout()
    save_fig(save_name)
    plt.close(fig)


def run_pretrain_comparison():
    shared = dict(n_steps=300, print_every=250, lr=1e-3, n_updates=8, seed=0)
    methods = {
        "MLE":       step_fkl_b,
        "REINFORCE": step_reinforce_pretrain,
        "Path":      step_b_path,
        "NPG-logit": step_npg_logit,
        "NPG-full":  step_npg_full,
    }
    histories = {}
    for name, fn in methods.items():
        _, h = train(fn, **shared)
        histories[name] = h
    plot_comparison(histories, b=B, save_name="pretrain_comparison.png")


run_pretrain_comparison()

