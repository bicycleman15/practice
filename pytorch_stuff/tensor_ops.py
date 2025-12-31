# tensor_indexing_practice.py
import torch


# -----------------------------
# Utilities
# -----------------------------
def seed_all(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def assert_close(a: torch.Tensor, b: torch.Tensor, msg: str = ""):
    if not torch.allclose(a, b):
        max_abs = (a - b).abs().max().item()
        raise AssertionError(f"{msg}\nNot close. max_abs_diff={max_abs}\nA={a}\nB={b}")


def run_test(name, fn):
    try:
        fn()
        print(f"[PASS] {name}")
    except NotImplementedError:
        print(f"[TODO ] {name} (NotImplementedError)")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        raise


# -----------------------------
# EXERCISES (fill these in)
# -----------------------------

# ---- torch.gather ----
def ex_gather_pick_one_per_row(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    x: (B, N)
    idx: (B,) each entry in [0, N)
    return y: (B,) where y[b] = x[b, idx[b]]
    """
    # TODO: implement using torch.gather (no advanced indexing x[torch.arange(B), idx])
    raise NotImplementedError


def ex_gather_3d(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    x: (B, T, C)
    idx: (B, T) indices in [0, C)
    return y: (B, T) where y[b,t] = x[b,t, idx[b,t]]
    """
    # TODO: implement using torch.gather
    raise NotImplementedError


def ex_gather_nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, T, V)
    targets: (B, T) in [0, V)
    return nll: (B, T) where nll[b,t] = -log_softmax(logits)[b,t,targets[b,t]]
    """
    # TODO: implement with log_softmax + gather
    raise NotImplementedError


# ---- torch.scatter / scatter_add ----
def ex_scatter_onehot(idx: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    idx: (B,) in [0, num_classes)
    return onehot: (B, num_classes)
    """
    # TODO: implement with scatter_
    raise NotImplementedError


def ex_scatter_add_hist(idx: torch.Tensor, w: torch.Tensor, num_bins: int) -> torch.Tensor:
    """
    idx: (B,) in [0, num_bins)
    w: (B,) weights
    return hist: (num_bins,) where hist[i] = sum_{b: idx[b]=i} w[b]
    """
    # TODO: implement with scatter_add_ (or scatter_ with reduce add if you want)
    raise NotImplementedError


def ex_scatter_reverse_gather(idx: torch.Tensor, y: torch.Tensor, n: int) -> torch.Tensor:
    """
    idx: (B, K) indices into dim=1 of size n
    y: (B, K) values
    return x_hat: (B, n) zeros except at idx positions.
      If duplicates in idx, SUM the corresponding y into x_hat (so use scatter_add).
    """
    # TODO: implement with scatter_add_
    raise NotImplementedError


def ex_scatter_bow(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    tokens: (B, T) in [0, vocab_size)
    return bow: (B, vocab_size) counts per vocab id (integer counts in float tensor OK)
    """
    # TODO: scatter_add counts
    raise NotImplementedError


# ---- torch.index_select ----
def ex_index_select_cols(x: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """
    x: (B, N)
    cols: (K,) in [0, N)
    return y: (B, K) selecting columns
    """
    # TODO: implement with torch.index_select
    raise NotImplementedError


def ex_index_select_timesteps(x: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
    """
    x: (B, T, C)
    t_idx: (K,) time indices in [0, T)
    return y: (B, K, C)
    """
    # TODO: implement with torch.index_select along time dimension
    raise NotImplementedError


# ---- broadcasting ----
def ex_broadcast_row_center(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, N)
    return x_centered: subtract per-row mean
    """
    # TODO: implement with broadcasting (use keepdim=True)
    raise NotImplementedError


def ex_broadcast_pairwise_dist2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: (M, D)
    b: (N, D)
    return dist2: (M, N) squared euclidean distances
    """
    # TODO: implement with broadcasting
    raise NotImplementedError


def ex_broadcast_per_head_scale(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    x: (B, H, T, D)
    scale: (H,)
    return scaled x where each head h is multiplied by scale[h]
    """
    # TODO: implement with broadcasting only (reshape scale)
    raise NotImplementedError


# -----------------------------
# TESTS (don’t edit)
# -----------------------------
def test_gather_pick_one_per_row():
    seed_all(0)
    B, N = 7, 11
    x = torch.randn(B, N)
    idx = torch.randint(0, N, (B,))
    y = ex_gather_pick_one_per_row(x, idx)
    y_ref = x[torch.arange(B), idx]
    assert y.shape == (B,)
    assert_close(y, y_ref, "gather_pick_one_per_row mismatch")


def test_gather_3d():
    seed_all(1)
    B, T, C = 4, 6, 9
    x = torch.randn(B, T, C)
    idx = torch.randint(0, C, (B, T))
    y = ex_gather_3d(x, idx)
    y_ref = x[torch.arange(B)[:, None], torch.arange(T)[None, :], idx]
    assert y.shape == (B, T)
    assert_close(y, y_ref, "gather_3d mismatch")


def test_gather_nll():
    seed_all(2)
    B, T, V = 3, 5, 13
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    nll = ex_gather_nll(logits, targets)

    logp = torch.log_softmax(logits, dim=-1)
    nll_ref = -logp.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    assert nll.shape == (B, T)
    assert_close(nll, nll_ref, "gather_nll mismatch")


def test_scatter_onehot():
    seed_all(3)
    B, C = 8, 10
    idx = torch.randint(0, C, (B,))
    oh = ex_scatter_onehot(idx, C)
    ref = torch.nn.functional.one_hot(idx, num_classes=C).to(dtype=oh.dtype)
    assert oh.shape == (B, C)
    assert_close(oh, ref, "scatter_onehot mismatch")


def test_scatter_add_hist():
    seed_all(4)
    B, N = 50, 7
    idx = torch.randint(0, N, (B,))
    w = torch.randn(B)
    hist = ex_scatter_add_hist(idx, w, N)

    ref = torch.zeros(N)
    ref.index_add_(0, idx, w)  # reference uses index_add_
    assert hist.shape == (N,)
    assert_close(hist, ref, "scatter_add_hist mismatch")


def test_scatter_reverse_gather():
    seed_all(5)
    B, n, K = 5, 12, 20
    idx = torch.randint(0, n, (B, K))
    y = torch.randn(B, K)
    x_hat = ex_scatter_reverse_gather(idx, y, n)

    ref = torch.zeros(B, n)
    ref.scatter_add_(dim=1, index=idx, src=y)
    assert x_hat.shape == (B, n)
    assert_close(x_hat, ref, "scatter_reverse_gather mismatch")


def test_scatter_bow():
    seed_all(6)
    B, T, V = 4, 30, 15
    tokens = torch.randint(0, V, (B, T))
    bow = ex_scatter_bow(tokens, V)

    # reference counts
    ref = torch.zeros(B, V)
    ones = torch.ones(B, T)
    ref.scatter_add_(dim=1, index=tokens, src=ones)
    assert bow.shape == (B, V)
    assert_close(bow, ref, "scatter_bow mismatch")


def test_index_select_cols():
    seed_all(7)
    B, N, K = 6, 10, 4
    x = torch.randn(B, N)
    cols = torch.tensor([0, 3, 7, 9])
    y = ex_index_select_cols(x, cols)
    ref = x[:, cols]
    assert y.shape == (B, K)
    assert_close(y, ref, "index_select_cols mismatch")


def test_index_select_timesteps():
    seed_all(8)
    B, T, C, K = 2, 9, 5, 3
    x = torch.randn(B, T, C)
    t_idx = torch.tensor([1, 4, 7])
    y = ex_index_select_timesteps(x, t_idx)
    ref = x[:, t_idx, :]
    assert y.shape == (B, K, C)
    assert_close(y, ref, "index_select_timesteps mismatch")


def test_broadcast_row_center():
    seed_all(9)
    B, N = 5, 7
    x = torch.randn(B, N)
    xc = ex_broadcast_row_center(x)
    ref = x - x.mean(dim=1, keepdim=True)
    assert xc.shape == (B, N)
    assert_close(xc, ref, "broadcast_row_center mismatch")


def test_broadcast_pairwise_dist2():
    seed_all(10)
    M, N, D = 4, 6, 3
    a = torch.randn(M, D)
    b = torch.randn(N, D)
    dist2 = ex_broadcast_pairwise_dist2(a, b)
    ref = ((a[:, None, :] - b[None, :, :]) ** 2).sum(dim=-1)
    assert dist2.shape == (M, N)
    assert_close(dist2, ref, "broadcast_pairwise_dist2 mismatch")


def test_broadcast_per_head_scale():
    seed_all(11)
    B, H, T, D = 2, 4, 3, 5
    x = torch.randn(B, H, T, D)
    scale = torch.randn(H)
    y = ex_broadcast_per_head_scale(x, scale)
    ref = x * scale.view(1, H, 1, 1)
    assert y.shape == (B, H, T, D)
    assert_close(y, ref, "broadcast_per_head_scale mismatch")


def main():
    tests = [
        ("gather_pick_one_per_row", test_gather_pick_one_per_row),
        ("gather_3d", test_gather_3d),
        ("gather_nll", test_gather_nll),
        ("scatter_onehot", test_scatter_onehot),
        ("scatter_add_hist", test_scatter_add_hist),
        ("scatter_reverse_gather", test_scatter_reverse_gather),
        ("scatter_bow", test_scatter_bow),
        ("index_select_cols", test_index_select_cols),
        ("index_select_timesteps", test_index_select_timesteps),
        ("broadcast_row_center", test_broadcast_row_center),
        ("broadcast_pairwise_dist2", test_broadcast_pairwise_dist2),
        ("broadcast_per_head_scale", test_broadcast_per_head_scale),
    ]
    for name, fn in tests:
        run_test(name, fn)


if __name__ == "__main__":
    main()
