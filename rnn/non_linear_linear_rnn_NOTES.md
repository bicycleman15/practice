# Non-linear RNN over linear chunks — notes

Design notes, experiments, and takeaways from building a Gated-DeltaNet-style
linear RNN with a switchable non-linearity `f(S)` applied only at chunk
boundaries. Companion to the code in `rnn/non_linear_linear_rnn*.py`.

The architecture was motivated by:
- NVlabs/GatedDeltaNet (per-token gated decay, delta-rule update)
- Luckyoriginal/deltanet (WY-form chunked parallel kernel)
- M²RNN, Mishra et al. 2026 (<https://arxiv.org/abs/2603.14360>) — the clearest
  modern statement of "linear RNNs can't state-track, non-linear RNNs can"
- hackmd.io/urUqHaXyTcGZaS05czIeKQ — the specific "non-linear-over-linear-chunks"
  proposal that kicked this off

---

## 1. The idea

Linear RNNs (Mamba, GDN, RetNet, DeltaNet, …) are fast because a chunked
associative scan is parallelizable inside each chunk. But they're stuck in TC⁰
and can't do state tracking.

Fully non-linear RNNs (LSTM, GRU, M²RNN) escape TC⁰ but break the chunked
parallel kernel.

Proposal: keep the update **linear inside a chunk** (so the WY kernel still
works) and apply an arbitrary non-linearity `f(S)` **once per chunk** at the
boundary:

```
S_{c+1} = f(decay_c · S_c + ΔS_c)
```

This buys a design knob: `chunk_size=1` is a fully non-linear RNN; `chunk_size=L`
is a fully linear RNN; in between, you trade expressivity for parallelism.

---

## 2. Architecture

All in `rnn/non_linear_linear_rnn.py`. The core pieces:

### `NonLinearLinearRNNConfig`
- `dim`, `num_heads`, `head_k_dim`, `head_v_dim`, `layers`, `vocab_size`
- `chunk_size` — the "tick rate" for the boundary non-linearity.
- `boundary_nonlin ∈ {identity, rmsnorm, tanh_res, gru}` — the `f(S)` to apply.
- `use_short_conv` — causal depthwise Conv1d on q/k/v (GDN trick).
- `decay_init ∈ {mild, strong}` — initial magnitude of the intra-chunk decay
  `gk`. Mild ≈ `-1e-4`/token; strong ≈ `-1e-2`/token.

### Boundary variants (on state `S ∈ R^{B×H×K×V}`)
| variant | formula | capability |
|---|---|---|
| `identity` | `S` | no-op; keeps the model a pure linear RNN |
| `rmsnorm` | `γ · S / RMS(S)` | normalizes magnitude; **destroys scale info** |
| `tanh_res` | `S + tanh(W·S)` | uniform non-linear smear; no selection |
| `gru` | `z⊙S + (1-z)⊙h`, `z = σ(·)`, `h = tanh(·)` | per-entry, content-dependent gating |

Only `gru` is both **content-aware** and **per-entry**: it can say "keep this
slot of S, flush that slot" based on what's currently in memory. Everything
else is either scalar-uniform or non-selective.

### Inter/intra-chunk math
- **Recurrent reference** (`delta_rule_recurrent`): token-by-token GDN update
  `S ← α S + β(v - Sk)k^T`, with `f(S)` applied every `chunk_size` tokens.
- **Chunked kernel** (`delta_rule_chunked`): WY form from Luckyoriginal
  + per-token cumulative-decay trick from GDN. Folds `gk` into `K_beta_up = β·r·k`
  and `K_down = k/r`, so the intra-chunk matrix inversion `T = (I + L)^{-1}`
  stays tractable. Then the inter-chunk loop over `N = L/C` chunks applies
  `f` exactly once per chunk.

Passes an `f=identity` equivalence test between the two at fp64:
`max out diff 4.4e-16`, `max state diff 3.3e-16`.

### Numerical safety
The decay `gk` can take extreme values after training; without care, `r = exp(G)`
underflows to 0 while `1/r` blows up to `inf` (→ NaN). Fix: clamp `G.clamp(min=-30)`
and `(-G).clamp(max=30)` in the chunked kernel. State is effectively gone at
those magnitudes anyway, so the clamp is lossless.

### Decay init
GDN's default init (`A ∈ [0, 16]`, `dt ∈ [1e-3, 1e-1]`) is calibrated for NLP
pretraining with long sequences. On sub-2k sequence retrieval benchmarks it
wipes the state out before it can be read, and also produces the NaN failure
mode above.

- `mild`: `A ∈ [0.1, 1.0]`, `dt ∈ [1e-4, 1e-3]` → `gk/token ~ -1e-4`.
  State retention over 64 tokens ≈ `0.994`.
- `strong`: `A ∈ [0.5, 4.0]`, `dt ∈ [1e-2, 1e-1]` → `gk/token ~ -1e-2`.
  State retention over 64 tokens ≈ `0.53`.

The model is still free to learn any decay afterwards — this only moves the
starting point.

---

## 3. Experiments

### 3.1 Smoke tests (`python -m rnn.non_linear_linear_rnn`)
- Shape test: forward returns `(B, L, V)` for all 4 × 2 (boundary × short_conv).
- Equivalence test: `chunked == recurrent` when `f=identity` (max diff ~4e-16 in fp64).
- 100-step overfitting on random tokens: all 4 × 2 variants drop from ~5.7 to ~0.2 nats.

### 3.2 Overfitting (`rnn/non_linear_linear_rnn_overfit.py`)
Single-batch overfit, 200 steps, all four boundary variants. All reach very low
loss on the memorization task → basic optimization works.

### 3.3 MQAR (Multi-Query Associative Recall, `rnn/non_linear_linear_rnn_mqar.py`)

Task: B K-V pairs followed by B queries drawn from those keys; loss only on
recall positions. With N=24 pairs, M=8 queries, vocab=64, chunk_size=16,
seq_len=64 (4 chunks), batch=32, 500 steps on MPS.

#### 4 boundary non-linearities, mild decay init
| variant | best acc | notes |
|---|---|---|
| random baseline | 1.56% | — |
| identity | 54.30% | pure linear baseline; solid |
| rmsnorm | 7.81% | **rmsnorm destroys the magnitude info that the KV outer products encode** |
| tanh_res | 30.86% | learns but slow; tanh squashes |
| **gru** | **72.27%** | content-aware per-entry gate consolidates well |

#### Adding a decay-init sweep
| variant | mild best | strong best | Δ |
|---|---|---|---|
| identity | 54.30% | **65.62%** | +11.3 |
| rmsnorm | 7.81% | 11.33% | +3.5 |
| tanh_res | 30.86% | 34.38% | +3.5 |
| **gru** | **72.27%** | 67.19% | **−5.1** |

Key finding: **strong intra-chunk decay helps non-gated boundaries a lot but
slightly hurts the GRU.** Mechanistically:

- Strong decay = non-selective, time-based forgetting. Useful if you have no
  other way to discard stale writes.
- GRU gate = selective, content-based forgetting. Already does the job.
- Stacking them = the decay throws away information the gate would have kept
  → they compete rather than compose.

So on short MQAR, the hackmd proposal's "strong within-chunk decay + non-linear
consolidation at boundary" doesn't stack as expected. The two mechanisms cover
the same function (selective memory retention) when the task fits in few chunks.

Conjecture: on *long* MQAR with many distractor chunks, strong decay filters
distractors and the GRU consolidates signal, and the stack wins. Not yet tested.

### 3.4 S_3 state tracking (`rnn/non_linear_linear_rnn_s3.py`)

Task: sequence of random `g_t ∈ S_3`, predict running product
`p_t = g_1 · … · g_t`. `S_3` is the smallest non-abelian group (6 elements),
so composition is not summarizable by any TC⁰ circuit in the general case.

Sanity-checked the Cayley table (identity laws, associativity, non-abelianness).

**First run** (`chunk_size=16`, `lr=3e-3`, `steps=1500`): all four variants
plateaued at ~20% accuracy — random is 16.67%. **Nothing learned the training
length**, so length generalization is moot.

Diagnosis:
1. Only 4 boundary fires per training sequence (`L_train=64 / chunk=16`) → the
   GRU boundary gets almost no gradient signal for learning the group action.
2. The delta rule's update is structurally a **right**-action
   `S(I − βkk^T) + βvk^T`; group-product tracking wants a **left**-action
   `S ← M(g_t) · S`. With a tiny state the two inductive biases fight; the
   model needs many steps to find an awkward workaround.
3. `lr=3e-3` is probably too hot for a 6-way classification task with tight
   info flow.

**Retuned** (`chunk_size=4`, `lr=1e-3`, `steps=4000`, `dim=96`,
`use_short_conv=True`) — not yet run to completion. Not the main point anyway;
the real takeaway is architectural:

> The delta-rule kernel's inductive bias is **key-value associative memory**,
> not **group-action state tracking**. Non-linear boundaries help at
> consolidation; they don't rescue the within-chunk kernel from being
> structurally wrong for this family of tasks.

This is what M²RNN calls out in its design: they replace the entire
recurrence with a non-linear matrix-matrix update every token, not just at
boundaries. That's the cleanest fix for state tracking.

---

## 4. What we learned (summary)

1. **Expressivity ladder.** Pure linear RNN ≺ linear-within-chunks +
   non-linear-at-boundary ≺ fully non-linear RNN (M²RNN). The chunk_size knob
   interpolates the middle row.

2. **Non-linearity choice matters, not just its presence.** At the boundary,
   rmsnorm is actively harmful for KV-memory tasks (normalizes away magnitude).
   tanh_res smears uniformly. GRU's per-entry, content-dependent gate is the
   only one that selectively consolidates memory. On MQAR, gru beat identity
   by +19 acc points; rmsnorm lost 46 points to identity.

3. **Gated DeltaNet vs. GRU-at-boundary is a real distinction, not cosmetic.**
   GDN's gates `α_t`, `β_t` are:
   - **input-dependent**, not state-dependent — memory has no say in its own update;
   - **scalar per head** — can't keep row i of S and flush row j;
   - **linear in S** — cannot mix S with itself non-linearly.
   The GRU boundary is state-dependent, per-entry, and non-linear. That's three
   orthogonal capabilities. GDN alone can't express any of them.

4. **Intra-chunk decay and boundary non-linearity partially substitute.**
   On MQAR, strong decay + linear boundary ≈ mild decay + GRU boundary. They
   don't stack cleanly because both are doing "selective memory retention" on
   a task whose working set fits in few chunks. The hackmd proposal's pitch
   ("fast forgetting + slow consolidation") probably needs a longer/harder
   task to show its full benefit.

5. **Delta-rule's right-action bias is the wrong prior for group tracking.**
   Even when the architecture is non-linear enough in principle (e.g., GRU
   at every boundary), the *within-chunk* update is still the KV-memory
   delta rule, and small state + right-action structure makes learning
   group-product composition very slow. This is the structural reason M²RNN
   replaces the whole recurrence rather than patching boundaries.

6. **Default GDN init is too aggressive for short retrieval-heavy sequences.**
   `A ∈ [0, 16]`, `dt ∈ [1e-3, 1e-1]` gives `gk/token ~ -1` at the extreme,
   which wipes 192-token state and causes `1/r → inf → NaN`. The mild init
   (`gk/token ~ -1e-4`) fixes both the learning failure and the NaN; a
   `clamp(-30, 30)` on `G` is a cheap belt-and-suspenders numerical guard.

---

## 5. Files

| file | what |
|---|---|
| `rnn/non_linear_linear_rnn.py` | config, boundary variants, short conv, chunked + recurrent kernels, block, LM, smoke tests |
| `rnn/non_linear_linear_rnn_overfit.py` | single-batch overfit sanity check across all variants |
| `rnn/non_linear_linear_rnn_mqar.py` | MQAR recall benchmark with sweep over `boundary_nonlin × decay_init` |
| `rnn/non_linear_linear_rnn_s3.py` | S_3 state-tracking length-generalization scaffold (WIP) |

---

## 6. Open threads / next steps

- **Finish the retuned S_3 run.** Expect `identity` to fit `L_train` but not
  generalize; `gru` with small chunk_size to generalize. If neither learns,
  the delta-rule bias is the limiting factor and the answer is "go to M²RNN".
- **Parity first.** `Z_2` parity is abelian and much easier than `S_3`. Good
  diagnostic: if a variant can't even do parity length-generalization,
  something is broken upstream of the expressivity story.
- **Long-MQAR stacking test.** Repeat the decay × boundary sweep on a long
  sequence with heavy distractors; check whether `gru + strong` overtakes
  `gru + mild` when the working set exceeds what the GRU gate alone can hold.
- **Inductive bias for group tracking.** The delta rule's right-action makes
  `S ← M·S` hard. A minimal fix: add a second update path that acts *left*,
  e.g. `S ← (I + β·u·v^T) S` (still linear, still chunkable via a slightly
  different WY derivation). Worth trying as a "group-biased delta" variant.
- **Scale up.** Everything here is 2-layer, `dim ≤ 128`, sequence ≤ 192. The
  findings are qualitative; magnitudes may shift at scale.

---

## 7. The `gru_input` failure on S_5, and the `m2rnn` fix

After moving to S_5 (`|S_5| = 120`, the smallest non-solvable group), the
`gru_input` boundary at `chunk_size=1` plateaued at ~2.4% accuracy
(random ≈ 0.83%, train loss ≈ log 120 = 4.79) — i.e. effectively did not
learn. This was at first surprising: `gru_input` at `chunk_size=1` is, on
paper, a fully non-linear input-conditioned RNN, which is universal. So what
goes wrong, and why does M²RNN succeed on the same task with seemingly the
same machinery?

### 7.1 The diff in one line

The candidate updates in the two recurrences look superficially identical
(`tanh(state-side + input-side)`), but the **shape of the input contribution
is qualitatively different**:

| | state-side (linear in S) | input-side | shape |
|---|---|---|---|
| `gru_input` | `S W_h` | `W_xh c` | broadcast across K → same offset on every K-row of S |
| `m2rnn` (this paper) | `H W` | `k_c v_c^T` | rank-1 over (K, V) → per-row offset, k_c selects which row |

That single structural difference is what makes M²RNN work and `gru_input`
fail on non-abelian state tracking.

#### Code: `gru_input` boundary — broadcast input, the BEFORE

`BoundaryNonlinearity.forward(self, S, c)` for `kind="gru_input"`:

```python
# c: (B, H, V) -> (B, H, 1, V) so it broadcasts over the K axis of S.
cWz = torch.einsum("bhv,hvu->bhu", c, self.W_xz)[:, :, None, :]
cWh = torch.einsum("bhv,hvu->bhu", c, self.W_xh)[:, :, None, :]
z = torch.sigmoid(self.b_z[None] + cWz + torch.einsum("bhkv,hvu->bhku", S, self.W_z))
h = torch.tanh(   self.b_h[None] + cWh + torch.einsum("bhkv,hvu->bhku", S, self.W_h))
return z * S + (1.0 - z) * h
```

Note the input contributions `cWz, cWh` are shape `(B, H, 1, V)` — the
`1` on the K axis is what broadcasts across rows. Every row of S sees the
same input offset.

#### Code: `m2rnn` boundary — rank-1 input, the AFTER (attempt 1)

`BoundaryNonlinearity.forward(self, S, c)` for `kind="m2rnn"`:

```python
# c: (B, H, K + V + 1) — split into k_c, v_c, f_raw.
K, V = self.head_k_dim, self.head_v_dim
k_c   = c[..., :K]                              # (B, H, K)
v_c   = c[..., K:K + V]                         # (B, H, V)
f_raw = c[..., -1]                              # (B, H)
f = torch.sigmoid(self.b_f + f_raw)             # (B, H)
KV = torch.einsum("bhk,bhv->bhkv", k_c, v_c)    # (B, H, K, V)  <-- rank-1 over (K, V)
SW = torch.einsum("bhkv,hvu->bhku", S, self.W)  # (B, H, K, V)
Z = torch.tanh(SW + KV)
f_b = f[:, :, None, None]                        # (B, H, 1, 1)
return f_b * S + (1.0 - f_b) * Z
```

The single line that matters is `KV = einsum("bhk,bhv->bhkv", k_c, v_c)`:
the input contribution is now shape `(B, H, K, V)` and is rank-1 across
both axes, not broadcast. Row `i` of S gets offset `k_{c,i} * v_c^T`, so
`k_c` selects the row, `v_c` the value.

### 7.2 Why "rank-1 over (K, V)" is the right primitive

State tracking on a finite group requires the recurrence to act, on each
step, as an *input-conditioned* linear map of the state — schematically
`S ← M(x_t) · S` where `M(x_t)` ranges over a generating set of the group.
A non-linear RNN that hopes to represent this needs the input to *select
where in the state to write*, not just to add a global bias.

- `k_c v_c^T` has K-axis structure: row `i` of the input increment is
  `k_{c,i} · v_c^T`. So `k_c` acts as an addressable index — different
  inputs activate different K-rows. This is exactly the
  associative-memory write primitive (`(key, value) -> S[key] = value`).
- `W_xh c` has no K-axis structure — it broadcasts to every row identically.
  No "addressing" possible. The optimizer cannot pose, let alone solve,
  the row-permutation problem.

This is not just an empirical observation — Mishra et al. (2026) prove
(Thm 1) that the M²RNN recurrence represents every function representable
by a non-linear vector-valued RNN, and demonstrate perfect S_5 length
generalization.

### 7.3 The `m2rnn` boundary variant

Implemented in `BoundaryNonlinearity` as `kind="m2rnn"`. Recurrence
(applied at chunk boundaries; reduces to literal M²RNN when `chunk_size=1`):

```
Z      = tanh(S W + k_c v_c^T)        # (B, H, K, V)
S_new  = f S + (1 - f) Z
```

with:
- `W ∈ R^{H×V×V}`: input-independent, identity-initialized. Init makes the
  recurrence at step 0 a pure saturating associative memory; W is learned.
- `(k_c, v_c, f_raw)`: split out of the per-chunk conditioning vector
  `c ∈ R^{B×H×K+V+1}`, projected from the chunk's last token by `c_proj`.
- `f = σ(b_f + f_raw) ∈ [0, 1]`: scalar per-head forget gate (we use a
  plain sigmoid; M²RNN uses `ψ(x) = (1 + e^(x+β))^(-α)` for a sharper
  parameterization, but it doesn't matter at this benchmark scale).

Differences vs the paper, all minor:

1. The forget gate is a sigmoid, not M²RNN's `ψ`.
2. We don't add the residual `w_r ⊙ v_t` to the output; the existing
   delta-rule output path already gives a residual via `o_intra + o_inter`.
3. We don't apply per-step gradient clipping on `S` during BPTT (M²RNN
   does); the global `clip_grad_norm_(1.0)` we already have seems to suffice.
4. We apply the recurrence at chunk boundaries; M²RNN applies it at every
   token. With `chunk_size=1` they coincide; with `chunk_size>1` we keep
   the linear delta-rule kernel for intra-chunk math and use M²RNN as the
   non-linear consolidator at the boundary.

### 7.4 Plumbing change: variant-specific boundary input dim

Boundaries that consume input now declare a per-head conditioning dim:

| variant | `boundary_input_per_head_dim` | what it carries |
|---|---|---|
| `gru_input` | `V` | broadcast offset |
| `m2rnn` | `K + V + 1` | `[k_c | v_c | f_raw]` flattened per head |

`NonLinearLinearRNNAttn.c_proj` now sizes its output to
`H * boundary_input_per_head_dim` rather than the previous hardcoded
`H * V`. Both `delta_rule_recurrent` and `delta_rule_chunked` are unchanged
(they treat `boundary_input` as opaque), and the `chunked == recurrent`
fp64 equivalence test passes for `m2rnn` at `max diff ~7e-16`.

### 7.5 First S_5 run with `m2rnn` boundary: also failed

Running the `m2rnn` boundary variant on S_5 at `chunk_size=1`, with the
rank-1 input fix in place, *also* plateaued at ~2.3% accuracy (loss 4.71,
random ≈ 0.83%). Same failure mode as `gru_input`. So rank-1-vs-broadcast
was necessary but not sufficient.

The reason is a structural one I missed: `m2rnn`-as-a-boundary is **not**
M²RNN. It is

    delta-rule kernel (rank-1 update via β k_t v_t^T)
       ↓
    m2rnn boundary  (rank-1 update via k_c v_c^T)

— *two* rank-1 outer-product writes per token, both functions of the same
input x_t but via *different* projections (`k_proj/v_proj` for the kernel,
`c_proj` for the boundary). The optimizer has to either (a) make one path
dormant and let the other carry the signal, or (b) coordinate the two paths
to write coherent values. Neither is encouraged by any inductive bias, and
the model just sits at random.

By contrast, the github reference M²RNN (open-lm-engine/accelerated-model-architectures)
has *one* rank-1 update per token, driven by a single set of projections
(`q, k, v, f` from one `input_projection` linear). That cleanness is what
makes optimization feasible.

#### Code: the broken hybrid, per token at chunk_size=1

What `linear_chunks/m2rnn` actually computes per token:

```python
# 1) delta-rule kernel (linear): writes one rank-1 outer product into S
S = exp(gk_t) * S
v_old = S^T @ k_t
S = S + k_t ⊗ (beta_t * (v_t - v_old))     # <-- rank-1 update #1, from k_proj/v_proj

# 2) m2rnn boundary (non-linear): writes ANOTHER rank-1 outer product on top
k_c, v_c, f_raw = split(c_proj(x_t))
S' = tanh(S @ W + k_c ⊗ v_c)               # <-- rank-1 update #2, from c_proj
S  = sigmoid(f_raw) * S + (1 - sigmoid(f_raw)) * S'
```

Two writes per token, two independent projection paths from the same x_t.
Nothing forces them to cooperate, so the optimizer just sits there.

### 7.6 The proper fix: M²RNN as a top-level architecture

Added `architecture: str = "linear_chunks" | "m2rnn"` to
`NonLinearLinearRNNConfig`. The block dispatches on it:

- `architecture="linear_chunks"` (default): the original delta-rule WY
  kernel + boundary nonlinearity. `chunk_size`, `boundary_nonlin`,
  `decay_init`, `allow_neg_eigval` are all honored.
- `architecture="m2rnn"`: the new `M2RNNAttn` module — pure M²RNN per-token
  recurrence with no delta-rule and no chunking. `chunk_size`,
  `boundary_nonlin`, `decay_init`, `allow_neg_eigval` are ignored.

`M2RNNAttn` follows the public reference torch implementation
(`forward_backward_torch` in op.py), with two minor deviations:

1. **W is identity-initialized** (paper §3.1, "Transition Matrix
   Initialization"; "identity initialization performs on par with
   orthogonal and we adopt it for all models"). The github code's
   `module.py.reset_parameters` uses `nn.init.normal_`, which seems
   inconsistent with the paper. With identity init at H_0=0 the recurrence
   reduces to H_1 = (1-f) tanh(k_1 v_1^T), a clean associative-memory write.
2. **f is sigmoided**. Github passes f raw; the paper uses the
   parameterized ψ(x) = (1+e^(x+β))^(-α). We just sigmoid for stability.

Recurrence (per token):

    cand = tanh(H_{t-1} W + k_t v_t^T)
    H_t  = f_t H_{t-1} + (1 - f_t) cand
    y_t  = q_t^T H_t

with `q, k, v, f` from independent linear projections (+ optional short
conv on q/k/v to match our linear_chunks setup), W as a learnable matrix
of shape (N, V, V) initialized to identity, and f as sigmoided per-token
per-head scalar.

#### Code: the AFTER (proper fix) — `M2RNNAttn.forward`

One coherent rank-1 update per token, no delta-rule preamble:

```python
q = self.q_proj(x); k = self.k_proj(x); v = self.v_proj(x)
if self.use_short_conv:
    q = self.q_conv(q); k = self.k_conv(k); v = self.v_conv(v)
q = q.view(B, L, N, K)
k = k.view(B, L, N, K)
v = v.view(B, L, N, V)
f = torch.sigmoid(self.f_proj(x))                 # (B, L, N)

kv = k.unsqueeze(-1) * v.unsqueeze(-2)            # (B, L, N, K, V)  <-- the only rank-1 write
W  = self.W.unsqueeze(0)                          # (1, N, V, V), identity-init

H = torch.zeros(B, N, K, V, ...)
ys = []
for t in range(L):
    f_t  = f[:, t, :, None, None]                 # (B, N, 1, 1)
    cand = torch.tanh(H @ W + kv[:, t])           # (B, N, K, V)
    H    = f_t * H + (1.0 - f_t) * cand           # gated update
    y_t  = (q[:, t].unsqueeze(-2) @ H).squeeze(-2)  # (B, N, V)
    ys.append(y_t)
y = torch.stack(ys, dim=1).flatten(-2, -1)         # (B, L, N*V)
out = self.o_proj(y)
```

Single set of projections (`q_proj, k_proj, v_proj, f_proj`), single rank-1
write per token (`k ⊗ v`), no delta-rule, no chunking. Matches the github
reference's `forward_backward_torch` line-for-line modulo the two paper-aligned
deviations (identity init on W, sigmoid on f).

This is the architecture to run on S_5; `linear_chunks/m2rnn` was the
wrong knob.

### 7.7 Recap of the failed-run progression

| run | result on S_5, train L=64 | reason it failed |
|---|---|---|
| `linear_chunks/gru_input`, cs=1 | ~2.4%, plateau | input contribution is broadcast across K, can't address rows; can't learn input-conditioned permutations |
| `linear_chunks/m2rnn`, cs=1 | ~2.3%, plateau | rank-1 input fixed, but delta-rule + m2rnn-boundary = two redundant rank-1 writes per token; optimizer can't disentangle |
| `architecture="m2rnn"`, true M²RNN | (running) | should work — single coherent rank-1 update per token, matches the reference architecture that demonstrably solves S_5 |


## 8. The `m2rnn_full_block` trap on state-tracking

After getting the recurrence right (Section 7), we still trailed the
reference badly on S_3 with matched capacity (dim=384, heads=12, lr=1e-3,
train L=16):

| | L=16 | L=32 | L=64 | L=128 |
|---|---|---|---|---|
| ours, full_block=True, identity W | 57.86% | 37.11% | 27.15% | 21.70% |
| ours, full_block=True, Xavier W   | 68.48% | 50.56% | 41.83% | 38.09% |
| ours, full_block=False, Xavier W  | **>97%** at 60% of training | — | — | — |
| ref_m2rnn                         | 99.29% | 93.32% | 64.92% | 43.38% |

Two changes closed the gap:

### 8.1 W init: identity → Xavier-normal `std=1/sqrt(V)`

Before (paper-style):
```python
W_init = torch.eye(V).unsqueeze(0).repeat(N, 1, 1)
self.W = nn.Parameter(W_init)
```

After (matches reference):
```python
self.W = nn.Parameter(torch.empty(N, V, V))
nn.init.normal_(self.W, mean=0.0, std=1.0 / math.sqrt(V))
```

Worth ~10 points at L=16. Same spectral-radius scale as identity at init,
but lets the model mix the V axis from step 0 instead of having to learn
deviations from a no-op.

### 8.2 The real culprit: `m2rnn_full_block`

The paper's full block (Eq. 20-21) wraps the M²RNN readout in three extra
operations:

```python
# What we had (full_block=True):
y = y + self.w_r[None, None, :, :] * v        # value residual
y = y.flatten(-2, -1)                         # (B, L, N*V)
g = F.silu(self.g_proj(x))                    # output gate, input-only
y = self.out_norm(y * g)                      # RMSNorm before o_proj
out = self.o_proj(y)

# What the open-source ref does (and what works for S_n):
y = y.flatten(-2, -1)
out = self.o_proj(y)
```

Each of the three "improvements" actively hurts state-tracking:

| component | math | why it hurts state-tracking |
|---|---|---|
| **output gate** | `y ← SiLU(W_g x_t) ⊙ y` | `g(x_t)` is a function of the **current** token only. The S_n target depends on the running product over **all** past tokens. Multiplying the readout by a per-step input-only gate scrambles the accumulated signal — the model has to fight to keep `g ≈ 1`. |
| **value residual** | `y ← y + w_r ⊙ v_t` | Injects the **current** `v_t` directly into the output, biasing the readout toward the latest token rather than the state `H_t`. For S_n the answer at `t` depends on the product `g_1 ⋯ g_t`; adding `v_t` is pure noise w.r.t. the target. |
| **RMSNorm before `o_proj`** | `y ← RMSNorm(y)` | Per-step input-independent magnitude rescaling of the readout. Compounds the above; the state's effective scale is reset each step. |

The recurrence itself was already correct. We just needed to stop
"improving" the readout. Fix:

```python
# rnn/non_linear_linear_rnn.py
m2rnn_full_block: bool = False   # was True
```

### 8.3 When *would* you want a value residual?

Useful when the **target depends heavily on the current token / recent
context** and the state is auxiliary background:

1. **Language modeling.** Next-token is overwhelmingly predicted by the
   last few tokens. Letting `v_t` skip the recurrent state gives a direct
   path for "what was just said"; the state only adds long-range context
   on top. RWKV-7, GLA, the M²RNN paper, and the "value residual learning"
   line all use it for LM.
2. **Compression-bottlenecked architectures.** Linear attention/RNNs
   compress unbounded history into a fixed-size `H ∈ ℝ^{K×V}`. Reading
   `q_t^T H_t` is lossy. If the prediction at `t` mostly needs `v_t`
   itself (copy, induction-head completion), forcing it through the state
   bottleneck wastes capacity. The residual is a cheap escape valve.
3. **Deep stacks.** Cross-layer value residuals ("Value Residual Learning
   For Alleviating Attention Concentration") give the first-layer `v` a
   direct path into deeper layers, mitigating the "deep layers forget what
   tokens were" problem.
4. **Translation / seq2seq.** Each output token aligns to a small input
   window; the current-token `v` is more informative than the running
   state.

Useless or harmful when the **state is the answer**:

- State tracking (S_n, parity, group word problems): the target is the
  running product, not anything intrinsic to `x_t`.
- Any task where `P(y_t | x_t) ≈ P(y_t)` (current token alone tells you
  almost nothing about the answer).

Rule of thumb:

```
P(y_t | x_t) ≈ P(y_t | x_1..x_t)  →  value residual helps  (LM, copy, retrieval)
P(y_t | x_t) ≈ P(y_t)             →  value residual hurts (S_n, parity, group products)
```

How much does the **current token alone** narrow the answer? High → free
win. Near-zero → dead weight at best, distraction at worst. The same
logic applies to the input-only output gate `g(x_t)`: fine when the
current token is informative, harmful when the state has to dominate.

### 8.4 Smaller deltas vs the reference (still in our impl, harmless)

For the record, after dropping `full_block`, we still differ from the
open-source reference in three minor ways:

- **ShortConv (Conv1d + SiLU) on q/k/v.** Local-window pre-mixing.
- **`f = sigmoid(f_proj(x))`** vs the reference's raw `f` (unbounded).
- **Per-step elementwise gradient clip on `H`** (matches the reference's
  `clip_gradients` STE helper, which we kept).

None of these blocked S_3; the curve tracks the reference once
`full_block=False`.

---

## 9. When does the boundary nonlinearity actually help?

After getting M²RNN to work, the natural follow-up was: forget the
architecture comparison — for the original proposal (linear delta-rule
kernel + nonlinearity at chunk boundaries), **on what task does the
nonlinearity earn its keep?** We swept three tasks.

### 9.1 S_3 state tracking — nonlinearity does not help

`linear_chunks`, `chunk_size=4`, sweep `boundary_nonlin ∈
{identity, rmsnorm, tanh_res, gru}`:

| variant | L=16 | L=32 | L=64 |
|---|---|---|---|
| **identity** | **~92%** | **~78%** | **~62%** |
| rmsnorm | ~85% | ~70% | ~55% |
| tanh_res | ~88% | ~72% | ~58% |
| gru | ~89% | ~74% | ~59% |

`identity` wins. S_n is a **regular** language: there's a finite-state
DFA of size `|S_n|` that solves it, and a linear-in-state RNN with
`state_dim ≥ |S_n|` can encode that DFA exactly as a transition-matrix
product (negative-eigenvalue β does the rest). Adding a nonlinearity at
the boundary just adds optimization noise — it can't expand the
expressivity class because the class is already enough.

### 9.2 Saturated counter — also DFA, also linear-solvable

To force a saturation event we built `non_linear_linear_rnn_satcount.py`:
counter that increments / decrements / holds with hard `[−K, K]` bounds.
Initially `K=8, L=16` was trivial (random walk almost never saturates →
task degenerates to running signed sum, which is *literally* linear).
Reduced to `K=2`. Result:

| variant | L=16 | L=32 | L=64 | L=128 |
|---|---|---|---|---|
| identity | 100% | 99.4% | 96.8% | 89.1% |
| rmsnorm | 99.9% | 99.0% | 95.5% | 87.3% |
| ... | | | | |

Still trivially solvable by linear. **A saturated counter with K states
is a DFA with K states.** The model represents each counter state as a
basis vector of S, and increments are 1-step shifts in that subspace —
all linear. The "saturation" nonlinearity that hits at the boundary is
already encodable as a transition matrix `M_+` whose `K`-th column is
zero. No `tanh` or `gru` at the boundary needed.

**Lesson:** any task expressible as a finite-state DFA is in principle
within reach of a linear-in-state RNN with enough state dim. "Looks
nonlinear" (clipping, saturation, modular arithmetic, group products) is
not the same as "needs a non-linear recurrence."

### 9.3 Dyck-2 stack-top — the real wall

Switched to a **strictly non-regular** task: predict the stack-top symbol
of a Dyck-2 (balanced `()`, `[]`) prefix. This is genuinely
context-free, not regular — no finite DFA can do it, because the stack
depth is unbounded.

To force the linear ceiling to bite, we:
- generated **balanced, deep** Dyck-2 with a triangular depth profile
  peaking at `L/4` (push prob `(L−t)/L`),
- shrank the model to `dim=64, heads=2, head_k=head_v=16` (matches
  reference),
- disabled `use_short_conv` (no local shortcut),
- reported and early-stopped on **post-pop accuracy** (positions where
  the stack-top must come from memory, not from the just-seen token).

With the reference M²RNN backend (per-token tanh nonlinearity, single
rank-1 write):

| | train pop_acc | L=16 | L=32 | L=64 | L=128 | L=256 | L=512 |
|---|---|---|---|---|---|---|---|
| ref_m2rnn (tanh, per-token) | 100% | 100% | 99% | 95% | 88% | 76% | **65%** |
| identity (linear) | 99% | 99% | 95% | 87% | 73% | 49% | **16%** |
| rmsnorm | 88% | 87% | 78% | 64% | 42% | 22% | 12% |
| tanh_res | 95% | 95% | 89% | 75% | 57% | 31% | 14% |
| gru | 96% | 95% | 88% | 73% | 53% | 28% | 13% |

Now the gap is unmistakable. Linear variants train fine in-distribution
(can fit a depth-`d` stack as long as `d ≤ state_dim`) but extrapolate
**below random** at `L=512`. M²RNN extrapolates gracefully — the
per-token `tanh(SW + kv^T)` actually compresses the stack rather than
just storing it.

Boundary nonlinearities (`tanh_res`, `gru`, `rmsnorm`) sit *between*
identity and m2rnn but closer to identity. Because the **within-chunk**
update is still linear, they can't break the regular-language ceiling
either; they just slightly improve the consolidation of what fits.

---

## 10. The big picture: linear vs non-linear RNNs

The whole arc of these notes is one observation:

> **Linear-in-state RNNs (with input-dependent A(x), B(x)) are exactly
> as expressive as DFAs. Non-linear RNNs are strictly more expressive
> (context-free → Turing-complete in the limit).**

### 10.1 The expressivity ladder

| class | gates depend on | per-step recurrence | formal power |
|---|---|---|---|
| LTI (S4 vanilla, RetNet) | nothing (constant) | `S ← A S + B u` | sub-regular (linear filter) |
| **Linear-in-state, input-gated** (Mamba, GLA, GDN, DeltaNet) | input `x_t` | `S ← A(x_t) S + B(x_t) u_t` | **regular (DFAs)** |
| **Non-linear in state** (RNN, LSTM, GRU, M²RNN) | input + state | `S ← σ(A(x_t) S + B(x_t) u_t)` | **context-free, ≥ regular**; Turing-complete in the limit (Siegelmann-Sontag '95) |

Two qualifiers matter:

1. **Input-dependent transitions are necessary.** A linear RNN with
   constant `A, B` (an LTI system) can't even XOR. The recurrences need
   `A(x_t), B(x_t)` to be functions of the current token to express any
   non-trivial DFA. This is why Mamba's "selective scan" is a big deal
   over S4-style time-invariance.
2. **The state nonlinearity is what breaks the DFA ceiling.** Composing
   `T` linear maps gives a linear map. Composing `T` non-linear maps
   gives an exponentially deeper non-linear function, which is the
   source of the extra expressivity.

So both knobs are real:
- input-dependent gating: LTI → regular.
- state nonlinearity:    regular → context-free / Turing-complete.

### 10.2 Why linear-in-state didn't die when we discovered it can't state-track

It looked for a while like Mamba/GLA/DeltaNet were "just DFAs" and
should be replaced by something more expressive. They weren't, because:

- **Most of language is regular at the local level.** Recent-context
  recall, induction heads, syntactic tagging, KV lookup — all DFA-fits.
  The CFG/CSL pieces of natural text are rare and the LM loss is
  dominated by local prediction.
- **Parallelism is huge.** Linear-in-state recurrences admit a parallel
  scan (associative composition of matrices). Non-linear recurrences
  are sequential. At training scale this is the difference between
  "1× hardware" and "20× hardware" for the same compute budget.
- **Gradient flow is clean.** Matrix products propagate gradients
  without the saturating-tanh problem; LSTM/GRU spent decades fighting
  what linear RNNs get for free.

The cost of going non-linear is real (gradient flow + parallelism), so
the practical design principle is **"use as little nonlinearity as you
can get away with."** Mamba/GLA/DeltaNet sit *just below* the regular
ceiling. M²RNN, vanilla RNN, LSTM sit *just above* it.

### 10.3 Where this project landed

- The original proposal (linear chunks + nonlinear boundary) is a
  reasonable interpolation point on the parallelism/expressivity
  trade-off, **but it doesn't escape the regular-language ceiling.**
  At `chunk_size=1` it does (it becomes a fully non-linear RNN), but
  then you've also lost the parallel-scan benefit.
- For S_n / counters / any DFA-fit task, plain linear delta-rule with
  enough state dim wins. The boundary nonlinearity is overhead.
- For genuinely non-regular tasks (deep Dyck-2, unbounded counting,
  context-free parsing), the **per-token state nonlinearity** in M²RNN
  is what does the work. A boundary-only nonlinearity, no matter how
  clever, can't replicate it because the within-chunk path is still
  linear.

### 10.4 Decision rule for picking an architecture

| if your task is... | use |
|---|---|
| dominated by local context (LM, copy, retrieval) | linear-in-state, input-gated. Add value residual + output gate to the readout. Mamba / GLA / DeltaNet. |
| a finite-state machine with bounded state (regular language, S_n, modular arithmetic, bounded counters) | linear-in-state, **no boundary nonlinearity needed**. State dim ≥ DFA size. |
| genuinely unbounded-state / context-free (deep Dyck, balanced parens, nested calls) | non-linear in state. M²RNN, LSTM, vanilla RNN. Pay the parallelism cost. |
| mixture (most realistic settings) | hybrid: linear-in-state mixer + occasional non-linear blocks (or Transformer attention layers, which are essentially per-token non-linear in S). |

The clean conceptual answer to the question that started this thread —
*are non-linear RNNs with input-dependent gating more powerful?* —
is **yes, strictly so**, but the gap only matters on tasks that genuinely
need it. Almost all of the practical wins come from input-dependent
gating alone (which is already a big jump from LTI), and the extra
state nonlinearity is a tool you reach for only when you can prove
your task is beyond regular.
