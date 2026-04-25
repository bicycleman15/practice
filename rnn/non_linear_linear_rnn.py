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
  - BoundaryNonlinearity            (identity / rmsnorm / tanh_res / gru / gru_input / m2rnn)
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
    boundary_nonlin: str = "rmsnorm"   # identity | rmsnorm | tanh_res | gru | gru_input | m2rnn
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

    # Top-level recurrence architecture.
    #   "linear_chunks":  Gated DeltaNet-style WY chunked kernel + boundary
    #                     non-linearity (the original of this file). Honors
    #                     `chunk_size`, `boundary_nonlin`, `decay_init`,
    #                     `allow_neg_eigval`.
    #   "m2rnn":          Pure M²RNN per-token recurrence (Mishra et al. 2026,
    #                     arXiv:2603.14360). Ignores chunk_size,
    #                     boundary_nonlin, decay_init, allow_neg_eigval.
    architecture: str = "linear_chunks"


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
    c (optional) has shape (B, H, D_c) — a per-chunk conditioning vector,
    where D_c = self.boundary_input_per_head_dim depends on the variant.
    Non-conditioned variants ignore `c`.

    --- gru_input vs m2rnn: the *shape of the input contribution* matters ----
    For input-conditioned variants the recurrence schematically looks like
        candidate = nonlin(  state-side  +  input-side  )
    The state-side term is a fixed linear map of S; the input-side term is
    where the input enters. Two qualitatively different choices:

      gru_input (broadcast offset):
          input-side =  W_xh c          ∈ R^{V}
          broadcast across the K axis of S.
          -> input adds the SAME offset to every K-row of S before the tanh.
          -> can't address a specific row → can't act as associative-memory
             write → can't represent input-conditioned permutations →
             provably can't track non-abelian groups (parity is fine; S_n is not).

      m2rnn (rank-1 outer product):
          input-side =  k_c ⊗ v_c        ∈ R^{K x V}
          rank-1 across BOTH axes; row i gets offset k_{c,i} · v_c^T.
          -> k_c selects WHICH K-row to write; v_c selects the value.
          -> this is the M²RNN recurrence (Mishra et al. 2026,
             arXiv:2603.14360, Eq. 19): Z = tanh(H W + k v^T),
             H_new = f H + (1-f) Z. They prove (Thm 1) it represents any
             non-linear vector-valued RNN, and demonstrate perfect S_5
             generalization. The rank-1-across-(K,V) input is exactly what
             gru_input lacks.
    -------------------------------------------------------------------------
    """

    #: variants that require the per-chunk conditioning `c` to be passed in.
    INPUT_CONDITIONED = ("gru_input", "m2rnn")

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
        elif kind == "m2rnn":
            # M²RNN recurrence (Mishra et al. 2026, arXiv:2603.14360, Eq. 19-20):
            #   Z = tanh(H W + k_c v_c^T)
            #   H_new = f H + (1 - f) Z
            # where W: (V, V) is input-INDEPENDENT (and identity-initialized),
            # k_c: (K,) and v_c: (V,) are functions of the chunk input,
            # and f ∈ [0, 1] is a scalar per-head forget gate.
            #
            # Key structural difference vs `gru_input`: the input contribution
            # is the rank-1 outer product k_c ⊗ v_c of shape (K, V), so the
            # input can address a SPECIFIC row of K (via k_c) with a value
            # (v_c) — i.e. the recurrence is an input-addressable associative
            # memory write. With `gru_input` the input is broadcast across K,
            # so it can't address rows.
            #
            # Identity init on W: the recurrence at init is
            #   Z = tanh(H + k_c v_c^T),  H_new = f H + (1-f) Z
            # i.e. a saturating associative memory. The model learns
            # deviations from identity. We use a plain sigmoid + per-head
            # bias for the forget gate; M²RNN uses a fancier
            # ψ(x) = (1 + e^(x+β))^(-α) parameterization, but a sigmoid is
            # close enough for this benchmark. b_f init = 0 -> f starts at
            # 0.5 (balanced); the model learns the equilibrium.
            W_init = torch.eye(head_v_dim).unsqueeze(0).repeat(num_heads, 1, 1)
            self.W   = nn.Parameter(W_init)
            self.b_f = nn.Parameter(torch.zeros(num_heads))
        else:
            raise ValueError(f"unknown boundary_nonlin kind: {kind}")

    @property
    def boundary_input_per_head_dim(self) -> int:
        """How many features per head the variant expects in the per-chunk
        conditioning vector `c`. Used by NonLinearLinearRNNAttn to size the
        c_proj projection. 0 for non-input-conditioned variants."""
        if self.kind == "gru_input":
            # c carries the V-dim broadcast offset.
            return self.head_v_dim
        if self.kind == "m2rnn":
            # c carries [k_c (K) | v_c (V) | f_raw (1)] flattened per head.
            return self.head_k_dim + self.head_v_dim + 1
        return 0

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
        if self.kind == "m2rnn":
            # c: (B, H, K + V + 1) — split into k_c, v_c, f_raw.
            assert c is not None, "m2rnn requires per-chunk conditioning c"
            K, V = self.head_k_dim, self.head_v_dim
            k_c   = c[..., :K]                              # (B, H, K)
            v_c   = c[..., K:K + V]                         # (B, H, V)
            f_raw = c[..., -1]                              # (B, H)
            f = torch.sigmoid(self.b_f + f_raw)             # (B, H)
            # Rank-1 input contribution: addressable associative-memory write.
            KV = torch.einsum("bhk,bhv->bhkv", k_c, v_c)    # (B, H, K, V)
            SW = torch.einsum("bhkv,hvu->bhku", S, self.W)  # (B, H, K, V)
            Z = torch.tanh(SW + KV)
            f_b = f[:, :, None, None]                        # (B, H, 1, 1)
            return f_b * S + (1.0 - f_b) * Z
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

        # For input-conditioned boundary variants we project the chunk's
        # "summary token" (last x in the chunk) into a per-head conditioning
        # vector c that is handed to f(S, c). The per-head dim is
        # variant-specific (gru_input: V; m2rnn: K+V+1; see
        # BoundaryNonlinearity.boundary_input_per_head_dim).
        self.needs_boundary_input = config.boundary_nonlin in BoundaryNonlinearity.INPUT_CONDITIONED
        if self.needs_boundary_input:
            self.boundary_input_per_head = self.boundary.boundary_input_per_head_dim
            self.c_proj = nn.Linear(
                config.dim, config.num_heads * self.boundary_input_per_head, bias=True,
            )

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
        # (last x in the chunk). Shape: (B, H, N, D_c) where D_c is the
        # per-head boundary input dim (variant-specific). Requires
        # L % chunk_size == 0, which the kernel also requires.
        boundary_input = None
        if self.needs_boundary_input:
            C = self.chunk_size
            assert L % C == 0, f"L={L} not divisible by chunk_size={C}"
            N = L // C
            x_last = x.reshape(B, N, C, -1)[:, :, -1, :]           # (B, N, D)
            c = self.c_proj(x_last)                                 # (B, N, H*D_c)
            boundary_input = einops.rearrange(c, "b n (h d) -> b h n d", h=H)

        out, S_final = delta_rule_chunked(
            q, k, v, beta, gk, S0, self.boundary, self.chunk_size,
            boundary_input=boundary_input,
        )
        out = einops.rearrange(out, "b h l v -> b l (h v)")
        out = self.o_proj(out)
        return out, S_final


# ---------------------------------------------------------------------------
# M²RNN: pure non-linear RNN, no delta-rule, no chunking
# ---------------------------------------------------------------------------


class M2RNNAttn(nn.Module):
    """Pure M²RNN per-token recurrence (Mishra et al. 2026, arXiv:2603.14360).

    Closely follows the public reference torch implementation in
    https://github.com/open-lm-engine/accelerated-model-architectures/blob/main/xma/layers/m2rnn/op.py
    (`forward_backward_torch`). Differences from the github code, all minor:
        - We initialize W with the identity (paper §3.1, "Transition Matrix
          Initialization") rather than `nn.init.normal_` (which is what the
          public `module.py.reset_parameters` does — paper says identity is
          on par with orthogonal and is what they actually use for models).
        - We apply `sigmoid` to the forget input so f ∈ [0, 1]. The github
          op.py passes `f` raw; presumably the user-side wraps it. We keep
          the gate well-defined.
        - We use the same `num_heads` for q, k, v, f, W (no multi-query /
          multi-value head sharing). This is sufficient for benchmarking.
        - We optionally apply the same depthwise short-conv on q, k, v as
          the linear-chunks path, so direct comparisons are apples-to-apples.

    Recurrence (torch reference, per token):
        x_t  = k_t v_t^T                                # (N, K, V) outer prod
        cand = tanh(H_{t-1} W + x_t)                     # (N, K, V)
        f_t  = sigmoid(W_f x_t + b_f)                    # (N,)
        H_t  = f_t H_{t-1} + (1 - f_t) cand              # (N, K, V)
        y_t  = q_t^T H_t                                  # (N, V)

    The whole loop is sequential — no chunked / parallel fast path. For our
    benchmarks (L ≤ 512) this is fine; the public repo's triton kernels
    are for production-scale training.
    """

    def __init__(self, config: NonLinearLinearRNNConfig):
        super().__init__()
        self.config = config
        D = config.dim
        N = config.num_heads
        K = config.head_k_dim
        V = config.head_v_dim
        self.N, self.K, self.V = N, K, V

        self.q_proj = nn.Linear(D, N * K, bias=False)
        self.k_proj = nn.Linear(D, N * K, bias=False)
        self.v_proj = nn.Linear(D, N * V, bias=False)
        # Forget-gate input projection. Output is per-head scalar; we apply
        # sigmoid in forward. b_f init = 0 -> f starts at 0.5 (balanced).
        self.f_proj = nn.Linear(D, N, bias=True)
        nn.init.zeros_(self.f_proj.bias)

        # Identity init on W (paper). With H_0 = 0 and W = I, the recurrence
        # at step 0 reduces to H_1 = (1 - f_1) tanh(k_1 v_1^T), i.e. a clean
        # associative-memory write. The model learns deviations from identity.
        W_init = torch.eye(V).unsqueeze(0).repeat(N, 1, 1)
        self.W = nn.Parameter(W_init)

        self.use_short_conv = config.use_short_conv
        if self.use_short_conv:
            self.q_conv = ShortConv(N * K, config.short_conv_kernel)
            self.k_conv = ShortConv(N * K, config.short_conv_kernel)
            self.v_conv = ShortConv(N * V, config.short_conv_kernel)

        self.o_proj = nn.Linear(N * V, D, bias=False)

    def forward(self, x, H0=None):
        B, L, D = x.shape
        N, K, V = self.N, self.K, self.V

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if self.use_short_conv:
            q = self.q_conv(q)
            k = self.k_conv(k)
            v = self.v_conv(v)
        q = q.view(B, L, N, K)
        k = k.view(B, L, N, K)
        v = v.view(B, L, N, V)

        # Sigmoid -> f ∈ [0, 1]. Github passes raw; we sigmoid for stability.
        f = torch.sigmoid(self.f_proj(x))                              # (B, L, N)

        # Rank-1 outer-product input contribution per token.
        kv = k.unsqueeze(-1) * v.unsqueeze(-2)                         # (B, L, N, K, V)

        if H0 is None:
            H = torch.zeros(B, N, K, V, device=x.device, dtype=q.dtype)
        else:
            H = H0

        W = self.W.unsqueeze(0)                                         # (1, N, V, V)

        ys = []
        for t in range(L):
            f_t = f[:, t, :, None, None]                                # (B, N, 1, 1)
            cand = torch.tanh(H @ W + kv[:, t])                         # (B, N, K, V)
            H = f_t * H + (1.0 - f_t) * cand
            y_t = (q[:, t].unsqueeze(-2) @ H).squeeze(-2)               # (B, N, V)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                                       # (B, L, N, V)
        y = y.flatten(-2, -1)                                            # (B, L, N*V)
        out = self.o_proj(y)
        return out, H


# ---------------------------------------------------------------------------
# block + LM
# ---------------------------------------------------------------------------


class NonLinearLinearRNNBlock(nn.Module):
    def __init__(self, config: NonLinearLinearRNNConfig):
        super().__init__()
        self.norm = nn.RMSNorm(config.dim)
        if config.architecture == "linear_chunks":
            self.attn = NonLinearLinearRNNAttn(config)
        elif config.architecture == "m2rnn":
            self.attn = M2RNNAttn(config)
        else:
            raise ValueError(f"unknown architecture: {config.architecture}")

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
        print(f"  linear_chunks  use_short_conv={use_conv}: logits {tuple(logits.shape)} ok")
    for use_conv in [True, False]:
        config = NonLinearLinearRNNConfig(
            dim=64, num_heads=2, head_k_dim=16, head_v_dim=16,
            layers=2, vocab_size=256,
            architecture="m2rnn", use_short_conv=use_conv,
        )
        model = NonLinearLinearRNNLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 64))
        logits = model(input_ids)
        assert logits.shape == (2, 64, config.vocab_size), logits.shape
        print(f"  m2rnn          use_short_conv={use_conv}: logits {tuple(logits.shape)} ok")


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

    # m2rnn: per-chunk c has shape (B, H, K + V + 1).
    torch.manual_seed(2)
    f = BoundaryNonlinearity("m2rnn", H, K, V).double()
    D_c = K + V + 1
    boundary_input = torch.randn(B, H, N, D_c, dtype=torch.float64)
    out_ref, S_ref = delta_rule_recurrent(q, k, v, beta, gk, S0, f, C, boundary_input=boundary_input)
    out_chu, S_chu = delta_rule_chunked(q, k, v, beta, gk, S0, f, C, boundary_input=boundary_input)
    print(f"  m2rnn       max out diff: {(out_ref - out_chu).abs().max().item():.2e}"
          f"   max S diff: {(S_ref - S_chu).abs().max().item():.2e}")
    assert torch.allclose(out_ref, out_chu, atol=1e-8, rtol=1e-6), "chunked disagrees with recurrent (m2rnn)"
    assert torch.allclose(S_ref, S_chu, atol=1e-8, rtol=1e-6)
    print("  pass")


def _test_training_all_variants():
    print("=== training smoke test (all boundary_nonlin x use_short_conv) ===")
    cases = [
        ("linear_chunks", k) for k in ["identity", "rmsnorm", "tanh_res", "gru", "gru_input", "m2rnn"]
    ] + [("m2rnn", "(n/a)")]
    for arch, kind in cases:
        for use_conv in [True, False]:
            torch.manual_seed(0)
            kwargs = dict(
                dim=64, num_heads=2, head_k_dim=16, head_v_dim=16,
                layers=2, vocab_size=256, use_short_conv=use_conv,
                architecture=arch,
            )
            if arch == "linear_chunks":
                kwargs.update(chunk_size=16, boundary_nonlin=kind)
            config = NonLinearLinearRNNConfig(**kwargs)
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

            assert not any(math.isnan(l) for l in losses), f"NaN loss: arch={arch} kind={kind} conv={use_conv}"
            assert losses[-1] < losses[0] + 1e-3, f"loss not decreasing: arch={arch} kind={kind} conv={use_conv}"
            tag = f"{arch}/{kind}" if arch == "linear_chunks" else arch
            loss_str = f"{losses[0]:.3f} -> ... -> {losses[-1]:.3f}"
            print(f"  {tag:<22s} conv={str(use_conv):<5s}: {loss_str}")


if __name__ == "__main__":
    _test_shapes()
    _test_equivalence()
    _test_training_all_variants()
    print("\nall tests passed")
