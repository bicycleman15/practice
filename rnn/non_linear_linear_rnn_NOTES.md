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
