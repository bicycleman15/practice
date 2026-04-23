"""
Non-linear RNN over linear chunks.

A Gated-DeltaNet-style linear recurrence where each chunk's intra-chunk math
stays linear/parallel (WY form from Luckyoriginal/deltanet, with per-token
gated decay from NVLabs/GatedDeltaNet), but a switchable non-linearity f(S)
is applied to the recurrent state at every chunk boundary.

    S_{c+1} = f(decay_c * S_c + delta_S_c)

See the hackmd writeup at https://hackmd.io/urUqHaXyTcGZaS05czIeKQ for the idea.

Layout of this file:
  - NonLinearLinearRNNConfig
  - BoundaryNonlinearity            (identity / rmsnorm / tanh_res / gru)
  - ShortConv                       (causal depthwise Conv1d + SiLU, with step mode)
  - delta_rule_recurrent            (reference, token-by-token)
  - delta_rule_chunked              (WY chunked form, f applied only at chunk edges)
  - NonLinearLinearRNNAttn          (projections + kernel)
  - NonLinearLinearRNNBlock         (pre-norm + attn + residual)
  - NonLinearLinearRNNLM            (embed + blocks + output head)
  - __main__ tests
"""

import math
from dataclasses import dataclass

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class NonLinearLinearRNNConfig:
    dim: int = 128
    num_heads: int = 4
    head_k_dim: int = 32
    head_v_dim: int = 32
    layers: int = 2
    vocab_size: int = 1024
    chunk_size: int = 64
    boundary_nonlin: str = "rmsnorm"   # identity | rmsnorm | tanh_res | gru
    use_short_conv: bool = True
    short_conv_kernel: int = 4

    # Controls the *initial* magnitude of the per-token decay gk inside a chunk.
    # gk ~ -exp(A_log) * softplus(gk_raw + dt_bias).
    # "mild"   -> dt ~ [1e-4, 1e-3], A ~ [0.1, 1.0]  -> gk/token ~ -1e-4
    # "strong" -> dt ~ [1e-2, 1e-1], A ~ [0.5, 4.0]  -> gk/token ~ -1e-2 to -1e-1
    # The model is free to learn any value afterwards; this only moves the start.
    decay_init: str = "mild"

    # If True, beta <- 2 * sigmoid(.) (range (0, 2)) so the per-token Householder
    # factor (I - beta k k^T) can have eigenvalues in (-1, 1) along k instead of
    # (0, 1). This is the "Unlocking State-Tracking via Negative Eigenvalues"
    # trick (Grazzi et al. 2024) and is required for DeltaNet to even solve
    # parity, let alone harder group word problems.
    allow_neg_eigval: bool = False


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------


def l2_norm(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


# ---------------------------------------------------------------------------
# boundary non-linearity on S: [B, H, K, V]
# ---------------------------------------------------------------------------


class BoundaryNonlinearity(nn.Module):
    """f(S, c) applied to the recurrent state at chunk boundaries.

    S has shape (B, H, K, V). All variants preserve the shape.
    c (optional) has shape (B, H, V) — a per-chunk conditioning vector, only
    used by input-conditioned variants (`gru_input`). Non-conditioned variants
    ignore `c`.
    """

    #: variants that require the per-chunk conditioning `c` to be passed in.
    INPUT_CONDITIONED = ("gru_input",)

    def __init__(self, kind: str, num_heads: int, head_k_dim: int, head_v_dim: int):
        super().__init__()
        self.kind = kind
        self.num_heads = num_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim

        if kind == "identity":
            pass
        elif kind == "rmsnorm":
            # RMSNorm over flattened (K*V) per head, with a per-head scalar scale.
            self.weight = nn.Parameter(torch.ones(num_heads, 1, 1))
            self.eps = 1e-6
        elif kind == "tanh_res":
            # S <- S + tanh(S @ W), W: (H, V, V) per head
            scale = 1.0 / math.sqrt(head_v_dim)
            self.W = nn.Parameter(torch.randn(num_heads, head_v_dim, head_v_dim) * scale)
        elif kind == "gru":
            # z = sigmoid(b_z + S @ W_z); h = tanh(b_h + S @ W_h); S_new = z*S + (1-z)*h
            scale = 1.0 / math.sqrt(head_v_dim)
            self.W_z = nn.Parameter(torch.randn(num_heads, head_v_dim, head_v_dim) * scale)
            self.W_h = nn.Parameter(torch.randn(num_heads, head_v_dim, head_v_dim) * scale)
            # Start z near 1 so the gate is almost pass-through at init.
            self.b_z = nn.Parameter(torch.full((num_heads, 1, 1), 3.0))
            self.b_h = nn.Parameter(torch.zeros(num_heads, 1, 1))
        elif kind == "gru_input":
            # Input-conditioned GRU. Same structure as `gru` but the gate and
            # candidate also depend on a per-chunk input vector c (B, H, V):
            #   z = sigmoid(b_z + c @ W_xz + S @ W_z)
            #   h = tanh(   b_h + c @ W_xh + S @ W_h)
            #   S_new = z * S + (1 - z) * h
            # Broadcasting puts c on the K axis: the same c influences every
            # row of S within a head. This is what the boundary needs in order
            # to implement an input-conditioned state update (e.g. group
            # multiplication by the chunk's "net element").
            #
            # Note: unlike the plain `gru` variant we do NOT bias the gate
            # toward pass-through (b_z = 3 there → z ≈ 0.95 at init). With
            # input conditioning the whole point is that the candidate h
            # should actually contribute to S_new, and a near-closed gate
            # attenuates gradient into c_proj / W_xz / W_xh by ~20x and into
            # the gate weights themselves by ~5x. Start the gate balanced.
            scale = 1.0 / math.sqrt(head_v_dim)
            self.W_z  = nn.Parameter(torch.randn(num_heads, head_v_dim, head_v_dim) * scale)
            self.W_h  = nn.Parameter(torch.randn(num_heads, head_v_dim, head_v_dim) * scale)
            self.W_xz = nn.Parameter(torch.randn(num_heads, head_v_dim, head_v_dim) * scale)
            self.W_xh = nn.Parameter(torch.randn(num_heads, head_v_dim, head_v_dim) * scale)
            # self.b_z  = nn.Parameter(torch.full((num_heads, 1, 1), 3.0))
            self.b_z  = nn.Parameter(torch.zeros(num_heads, 1, 1))
            self.b_h  = nn.Parameter(torch.zeros(num_heads, 1, 1))
        else:
            raise ValueError(f"unknown boundary_nonlin kind: {kind}")

    def forward(self, S, c=None):
        if self.kind == "identity":
            return S
        if self.kind == "rmsnorm":
            rms = S.pow(2).mean(dim=(-2, -1), keepdim=True).clamp_min(self.eps).sqrt()
            return (S / rms) * self.weight[None]  # (1, H, 1, 1) broadcast
        if self.kind == "tanh_res":
            SW = torch.einsum("bhkv,hvu->bhku", S, self.W)
            return S + torch.tanh(SW)
        if self.kind == "gru":
            z = torch.sigmoid(self.b_z[None] + torch.einsum("bhkv,hvu->bhku", S, self.W_z))
            h = torch.tanh(self.b_h[None] + torch.einsum("bhkv,hvu->bhku", S, self.W_h))
            return z * S + (1.0 - z) * h
        if self.kind == "gru_input":
            assert c is not None, "gru_input requires per-chunk conditioning c"
            # c: (B, H, V) -> (B, H, 1, V) so it broadcasts over the K axis of S.
            cWz = torch.einsum("bhv,hvu->bhu", c, self.W_xz)[:, :, None, :]
            cWh = torch.einsum("bhv,hvu->bhu", c, self.W_xh)[:, :, None, :]
            z = torch.sigmoid(self.b_z[None] + cWz + torch.einsum("bhkv,hvu->bhku", S, self.W_z))
            h = torch.tanh(   self.b_h[None] + cWh + torch.einsum("bhkv,hvu->bhku", S, self.W_h))
            return z * S + (1.0 - z) * h
        raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# short conv on q / k / v (optional, toggled by config.use_short_conv)
# ---------------------------------------------------------------------------


class ShortConv(nn.Module):
    """Causal depthwise Conv1d followed by SiLU.

    forward(x): x of shape (B, L, D) -> (B, L, D). Causal via tail-padding truncation.
    step(x_t, state): token-by-token path, returns (y_t, new_state).
    """

    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            groups=dim,
            padding=kernel_size - 1,
            bias=False,
        )

    def forward(self, x):
        B, L, D = x.shape
        y = self.conv(x.transpose(1, 2))     # (B, D, L + K - 1)
        y = y[..., :L]                        # drop tail pad for causality
        return F.silu(y.transpose(1, 2))

    def step(self, x_t, state):
        """x_t: (B, D). state: (B, D, K-1). Returns (y_t, new_state)."""
        # window holds the K most recent tokens: [state | x_t]
        window = torch.cat([state, x_t.unsqueeze(-1)], dim=-1)  # (B, D, K)
        w = self.conv.weight.squeeze(1)                           # (D, K)
        b = self.conv.bias                                        # (D,)
        y = (window * w[None]).sum(dim=-1) + b                    # (B, D)
        new_state = window[:, :, 1:]                              # shift left by one
        return F.silu(y), new_state

    def init_state(self, batch_size, device, dtype=None):
        return torch.zeros(
            batch_size, self.dim, self.kernel_size - 1,
            device=device, dtype=dtype or torch.float32,
        )


# ---------------------------------------------------------------------------
# recurrent reference (ground-truth simulator)
# ---------------------------------------------------------------------------


def delta_rule_recurrent(q, k, v, beta, gk, S0, f, chunk_size, boundary_input=None):
    """Token-by-token gated delta rule with f(S) at every `chunk_size`-th token.

    Shapes:
        q, k: (B, H, L, K)
        v:    (B, H, L, V)
        beta: (B, H, L)         in (0, 1) typically
        gk:   (B, H, L)         per-token log-decay, typically negative
        S0:   (B, H, K, V)      initial state
        boundary_input: (B, H, N, V) or None   per-chunk conditioning for f
    Returns:
        out:      (B, H, L, V)
        S_final:  (B, H, K, V)

    Per-token update (see Luckyoriginal delta_rule_recurrent_step + GDN decay):
        S   <- exp(gk_t) * S
        v_old = S^T k_t
        dv    = beta_t * (v_t - v_old)
        S   <- S + k_t (outer) dv
        o_t  = S^T q_t
    And f(S, c_i) is invoked whenever (t + 1) % chunk_size == 0, with
    c_i = boundary_input[:, :, i] if boundary_input is not None else None,
    where i = t // chunk_size is the index of the chunk that just ended.
    """
    B, H, L, K = q.shape
    V = v.shape[-1]
    S = S0
    outputs = []
    for t in range(L):
        q_t = q[:, :, t, :]
        k_t = k[:, :, t, :]
        v_t = v[:, :, t, :]
        beta_t = beta[:, :, t]
        gk_t = gk[:, :, t]

        S = torch.exp(gk_t)[..., None, None] * S
        v_old = torch.einsum("bhkv,bhk->bhv", S, k_t)
        dv = beta_t[..., None] * (v_t - v_old)
        S = S + torch.einsum("bhk,bhv->bhkv", k_t, dv)
        o_t = torch.einsum("bhkv,bhk->bhv", S, q_t)
        outputs.append(o_t)

        if (t + 1) % chunk_size == 0:
            c_i = boundary_input[:, :, t // chunk_size] if boundary_input is not None else None
            S = f(S, c_i)

    out = torch.stack(outputs, dim=2)  # (B, H, L, V)
    return out, S


# ---------------------------------------------------------------------------
# chunked WY kernel with per-token gated decay
# ---------------------------------------------------------------------------


def delta_rule_chunked(q, k, v, beta, gk, S0, f, chunk_size, boundary_input=None):
    """Chunked gated delta rule. Intra-chunk is linear/parallel (WY form);
    f(S) is applied only in the inter-chunk transition.

    Using the cumulative-log-decay trick (FLA-style):
        G(s) = sum_{s' <= s} gk_{s'}
        r(s) = exp(G(s))
        K_up   = r * k          (row-scaled)
        K_down = k / r          (row-scaled)
        Q_up   = r * q
        K_beta_up = beta * K_up

        (I + strict_lower(K_beta_up K_down^T)) U = V_beta - K_beta_up S_prev
        W_tilde = T K_beta_up,  U_tilde = T V_beta   where T = (I + L)^{-1}
        U       = U_tilde - W_tilde S_prev
        o_inter = Q_up S_prev
        o_intra = tril(Q_up K_down^T) U
        S_new   = r_total * S_prev + r_total * K_down^T U
        S       = f(S_new)   <-- the only non-linear step

    Requires L % chunk_size == 0.
    """
    B, H, L, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    assert L % C == 0, f"L={L} not divisible by chunk_size={C}"
    N = L // C

    q = q.reshape(B, H, N, C, K)
    k = k.reshape(B, H, N, C, K)
    v = v.reshape(B, H, N, C, V)
    beta = beta.reshape(B, H, N, C)
    gk = gk.reshape(B, H, N, C)

    G = gk.cumsum(dim=-1)              # (B, H, N, C)
    # Clamp to avoid inf/NaN when gk is extreme (state fully decayed anyway).
    r = G.clamp(min=-30.0).exp()       # (B, H, N, C)
    r_inv = (-G).clamp(max=30.0).exp() # numerically safer than 1/r when r is small
    r_total = r[..., -1:]              # (B, H, N, 1)

    r_       = r.unsqueeze(-1)         # (B, H, N, C, 1)
    r_inv_   = r_inv.unsqueeze(-1)
    beta_    = beta.unsqueeze(-1)

    K_beta_up = beta_ * r_ * k          # (B, H, N, C, K)
    K_down    = k * r_inv_              # (B, H, N, C, K)
    V_beta    = beta_ * v               # (B, H, N, C, V)
    Q_up      = r_ * q                  # (B, H, N, C, K)

    # Intra-chunk Gram. Keep strict lower only; set upper-incl-diagonal to 0.
    M = K_beta_up @ K_down.transpose(-1, -2)                # (B, H, N, C, C)
    upper_incl_diag = torch.triu(torch.ones(C, C, device=q.device, dtype=torch.bool), diagonal=0)
    Lmat = M.masked_fill(upper_incl_diag, 0)

    # T = (I + Lmat)^{-1} via unit-triangular solve.
    eye = torch.eye(C, device=q.device, dtype=q.dtype).expand(B, H, N, C, C)
    T = torch.linalg.solve_triangular(eye + Lmat, eye, upper=False, unitriangular=True)

    W_tilde = T @ K_beta_up                                  # (B, H, N, C, K)
    U_tilde = T @ V_beta                                      # (B, H, N, C, V)

    # Intra-chunk output kernel: Q_up K_down^T, strict upper zeroed (lower incl diag).
    A = Q_up @ K_down.transpose(-1, -2)                      # (B, H, N, C, C)
    strict_upper = torch.triu(torch.ones(C, C, device=q.device, dtype=torch.bool), diagonal=1)
    A = A.masked_fill(strict_upper, 0)

    # Sequential inter-chunk loop; f(S) is the only non-linearity.
    S = S0
    outs = []
    for i in range(N):
        Ui = U_tilde[:, :, i] - W_tilde[:, :, i] @ S         # (B, H, C, V)
        o_inter = Q_up[:, :, i] @ S                           # (B, H, C, V)
        o_intra = A[:, :, i] @ Ui                             # (B, H, C, V)
        outs.append(o_inter + o_intra)

        rC = r_total[:, :, i].unsqueeze(-1)                   # (B, H, 1, 1)
        S_new = rC * S + rC * (K_down[:, :, i].transpose(-1, -2) @ Ui)
        c_i = boundary_input[:, :, i] if boundary_input is not None else None
        S = f(S_new, c_i)

    out = torch.cat(outs, dim=2)  # (B, H, L, V)
    return out, S


# ---------------------------------------------------------------------------
# attn module: projections + short conv + kernel
# ---------------------------------------------------------------------------


class NonLinearLinearRNNAttn(nn.Module):
    def __init__(self, config: NonLinearLinearRNNConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim
        self.chunk_size = config.chunk_size

        k_total = config.num_heads * config.head_k_dim
        v_total = config.num_heads * config.head_v_dim

        self.q_proj = nn.Linear(config.dim, k_total, bias=False)
        self.k_proj = nn.Linear(config.dim, k_total, bias=False)
        self.v_proj = nn.Linear(config.dim, v_total, bias=False)
        self.beta_proj = nn.Linear(config.dim, config.num_heads, bias=True)
        self.gk_proj = nn.Linear(config.dim, config.num_heads, bias=True)
        self.o_proj = nn.Linear(v_total, config.dim, bias=False)

        self.use_short_conv = config.use_short_conv
        if self.use_short_conv:
            self.q_conv = ShortConv(k_total, config.short_conv_kernel)
            self.k_conv = ShortConv(k_total, config.short_conv_kernel)
            self.v_conv = ShortConv(v_total, config.short_conv_kernel)

        self.boundary = BoundaryNonlinearity(
            config.boundary_nonlin,
            config.num_heads,
            config.head_k_dim,
            config.head_v_dim,
        )

        # For input-conditioned boundary variants we also project the chunk's
        # "summary token" (last x in the chunk) into a per-head, V-dim vector c
        # that is handed to f(S, c). Built only when needed.
        self.needs_boundary_input = config.boundary_nonlin in BoundaryNonlinearity.INPUT_CONDITIONED
        if self.needs_boundary_input:
            self.c_proj = nn.Linear(config.dim, v_total, bias=True)

        # GDN-style log-decay init: gk = -exp(A_log) * softplus(gk_raw + dt_bias).
        # "mild":   gk/token ~ -1e-4  (state preserved long after chunk end)
        # "strong": gk/token ~ -1e-2  (state decays noticeably within a chunk)
        if config.decay_init == "mild":
            A_lo, A_hi = math.log(0.1), math.log(1.0)
            dt_min, dt_max = 1e-4, 1e-3
        elif config.decay_init == "strong":
            A_lo, A_hi = math.log(0.5), math.log(4.0)
            dt_min, dt_max = 1e-2, 1e-1
        else:
            raise ValueError(f"unknown decay_init: {config.decay_init!r}")
        A = torch.empty(config.num_heads).uniform_(A_lo, A_hi)
        self.A_log = nn.Parameter(A)
        dt = torch.exp(
            torch.rand(config.num_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=1e-5)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
        self.dt_bias = nn.Parameter(inv_dt)

    def forward(self, x, S0=None):
        B, L, _ = x.shape
        H = self.num_heads
        K = self.head_k_dim
        V = self.head_v_dim

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.use_short_conv:
            q = self.q_conv(q)
            k = self.k_conv(k)
            v = self.v_conv(v)

        q = einops.rearrange(q, "b l (h d) -> b h l d", h=H)
        k = einops.rearrange(k, "b l (h d) -> b h l d", h=H)
        v = einops.rearrange(v, "b l (h d) -> b h l d", h=H)

        q = l2_norm(q)
        k = l2_norm(k)

        beta = torch.sigmoid(self.beta_proj(x))                   # (B, L, H)
        if self.config.allow_neg_eigval:
            # beta in (0, 2) -> eigenvalue of (I - beta k k^T) along k in (-1, 1)
            beta = 2.0 * beta
        beta = einops.rearrange(beta, "b l h -> b h l")

        gk_raw = self.gk_proj(x)                                   # (B, L, H)
        gk = -torch.exp(self.A_log) * F.softplus(gk_raw + self.dt_bias)
        gk = einops.rearrange(gk, "b l h -> b h l")

        if S0 is None:
            S0 = torch.zeros(B, H, K, V, device=x.device, dtype=q.dtype)

        # Build per-chunk boundary input from the "summary token" of each chunk
        # (last x in the chunk). Shape: (B, H, N, V). Requires L % chunk_size == 0,
        # which the kernel also requires.
        boundary_input = None
        if self.needs_boundary_input:
            C = self.chunk_size
            assert L % C == 0, f"L={L} not divisible by chunk_size={C}"
            N = L // C
            x_last = x.reshape(B, N, C, -1)[:, :, -1, :]           # (B, N, D)
            c = self.c_proj(x_last)                                 # (B, N, H*V)
            boundary_input = einops.rearrange(c, "b n (h v) -> b h n v", h=H)

        out, S_final = delta_rule_chunked(
            q, k, v, beta, gk, S0, self.boundary, self.chunk_size,
            boundary_input=boundary_input,
        )
        out = einops.rearrange(out, "b h l v -> b l (h v)")
        out = self.o_proj(out)
        return out, S_final


# ---------------------------------------------------------------------------
# block + LM
# ---------------------------------------------------------------------------


class NonLinearLinearRNNBlock(nn.Module):
    def __init__(self, config: NonLinearLinearRNNConfig):
        super().__init__()
        self.norm = nn.RMSNorm(config.dim)
        self.attn = NonLinearLinearRNNAttn(config)

    def forward(self, x):
        out, _ = self.attn(self.norm(x))
        return x + out


class NonLinearLinearRNNLM(nn.Module):
    def __init__(self, config: NonLinearLinearRNNConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([NonLinearLinearRNNBlock(config) for _ in range(config.layers)])
        self.norm = nn.RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.output(x)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def _test_shapes():
    print("=== shape test ===")
    for use_conv in [True, False]:
        config = NonLinearLinearRNNConfig(
            dim=64, num_heads=2, head_k_dim=16, head_v_dim=16,
            layers=2, vocab_size=256, chunk_size=16,
            boundary_nonlin="rmsnorm", use_short_conv=use_conv,
        )
        model = NonLinearLinearRNNLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 128))
        logits = model(input_ids)
        assert logits.shape == (2, 128, config.vocab_size), logits.shape
        print(f"  use_short_conv={use_conv}: logits {tuple(logits.shape)} ok")


def _test_equivalence():
    print("=== equivalence test: chunked vs recurrent, fp64 ===")
    torch.manual_seed(0)
    B, H, L, K, V, C = 2, 2, 32, 8, 8, 8
    N = L // C

    q = l2_norm(torch.randn(B, H, L, K, dtype=torch.float64) * 0.5)
    k = l2_norm(torch.randn(B, H, L, K, dtype=torch.float64) * 0.5)
    v = torch.randn(B, H, L, V, dtype=torch.float64)
    beta = torch.sigmoid(torch.randn(B, H, L, dtype=torch.float64))
    # Mild negative decay so r stays in a sane range for the reference check.
    gk = -F.softplus(torch.randn(B, H, L, dtype=torch.float64)) * 0.1
    S0 = torch.zeros(B, H, K, V, dtype=torch.float64)

    # identity: no conditioning needed.
    f = BoundaryNonlinearity("identity", H, K, V).double()
    out_ref, S_ref = delta_rule_recurrent(q, k, v, beta, gk, S0, f, C)
    out_chu, S_chu = delta_rule_chunked(q, k, v, beta, gk, S0, f, C)
    print(f"  identity    max out diff: {(out_ref - out_chu).abs().max().item():.2e}"
          f"   max S diff: {(S_ref - S_chu).abs().max().item():.2e}")
    assert torch.allclose(out_ref, out_chu, atol=1e-8, rtol=1e-6), "chunked disagrees with recurrent (identity)"
    assert torch.allclose(S_ref, S_chu, atol=1e-8, rtol=1e-6)

    # gru_input: same random per-chunk conditioning in both kernels.
    torch.manual_seed(1)
    f = BoundaryNonlinearity("gru_input", H, K, V).double()
    boundary_input = torch.randn(B, H, N, V, dtype=torch.float64)
    out_ref, S_ref = delta_rule_recurrent(q, k, v, beta, gk, S0, f, C, boundary_input=boundary_input)
    out_chu, S_chu = delta_rule_chunked(q, k, v, beta, gk, S0, f, C, boundary_input=boundary_input)
    print(f"  gru_input   max out diff: {(out_ref - out_chu).abs().max().item():.2e}"
          f"   max S diff: {(S_ref - S_chu).abs().max().item():.2e}")
    assert torch.allclose(out_ref, out_chu, atol=1e-8, rtol=1e-6), "chunked disagrees with recurrent (gru_input)"
    assert torch.allclose(S_ref, S_chu, atol=1e-8, rtol=1e-6)
    print("  pass")


def _test_training_all_variants():
    print("=== training smoke test (all boundary_nonlin x use_short_conv) ===")
    for kind in ["identity", "rmsnorm", "tanh_res", "gru", "gru_input"]:
        for use_conv in [True, False]:
            torch.manual_seed(0)
            config = NonLinearLinearRNNConfig(
                dim=64, num_heads=2, head_k_dim=16, head_v_dim=16,
                layers=2, vocab_size=256, chunk_size=16,
                boundary_nonlin=kind, use_short_conv=use_conv,
            )
            model = NonLinearLinearRNNLM(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            input_ids = torch.randint(0, config.vocab_size, (2, 64))

            losses = []
            for _ in range(100):
                optimizer.zero_grad()
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), input_ids.view(-1))
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            assert not any(math.isnan(l) for l in losses), f"NaN loss: kind={kind} conv={use_conv}"
            assert losses[-1] < losses[0] + 1e-3, f"loss not decreasing: kind={kind} conv={use_conv}"
            loss_str = " -> ".join(f"{l:.3f}" for l in losses)
            print(f"  {kind:<10s} conv={str(use_conv):<5s}: {loss_str}")


if __name__ == "__main__":
    _test_shapes()
    _test_equivalence()
    _test_training_all_variants()
    print("\nall tests passed")
