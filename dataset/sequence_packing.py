"""
A. Batch collation + padding + masks

Write a collate_fn that takes a list of (tokens, length) and returns:
    x_padded: [B, T], y_padded: [B, T] (shifted next-token labels)
    lengths: [B]
    pad_mask: [B, T] or attn_mask-style boolean mask
"""


import torch

def pad(x, length, v):
    pad_tensor = torch.full((length,), v, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad_tensor])

def collate_fn(batch, pad_token=0, ignore_target=-100):
    
    # calculate the max seqlen 
    max_length = max(x for _, x in batch)

    input_ids = list()
    targets = list()
    lengths = list()
    pad_mask = list()

    for ids, l in batch:
        ids = torch.tensor(ids, dtype=torch.long)
        t = ids[1:].clone()

        # pad till max_length
        ids = pad(ids, max_length - ids.shape[0], v=pad_token).unsqueeze(0)
        t = pad(t, max_length - t.shape[0], v=ignore_target).unsqueeze(0)

        input_ids.append(ids)
        targets.append(t)
        lengths.append(l)
        pad_mask.append((ids != pad_token).bool().unsqueeze(0))

    input_ids = torch.cat(input_ids, dim=0)
    targets = torch.cat(targets, dim=0)
    pad_mask = torch.cat(pad_mask, dim=0)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return input_ids, targets, lengths, pad_mask

def collate_fn_prealloc(batch, pad_token=0, ignore_target=-100):
    
    # calculate the max seqlen
    max_length = max(x for _, x in batch)

    input_ids = torch.full((len(batch), max_length), pad_token, dtype=torch.long)
    targets = torch.full((len(batch), max_length), ignore_target, dtype=torch.long)
    lengths = list()
    pad_mask = torch.full((len(batch), max_length), False, dtype=torch.bool)

    for i, (ids, l) in enumerate(batch):
        ids = torch.tensor(ids, dtype=torch.long)
        t = ids[1:].clone()

        cur_len = ids.shape[0]

        input_ids[i, :cur_len] = ids
        targets[i, :cur_len-1] = t
        pad_mask[i, :cur_len] = True

        # pad till max_length
        lengths.append(l)

    # input_ids = torch.cat(input_ids, dim=0)
    # targets = torch.cat(targets, dim=0)
    # pad_mask = torch.cat(pad_mask, dim=0)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return input_ids, targets, lengths, pad_mask


if __name__ == "__main__":
    # Simple test for collate_fn
    batch = [
        (torch.tensor([1, 2, 3, 4, 5]), 5),
        (torch.tensor([6, 7, 8]), 3),
        (torch.tensor([9, 10]), 2),
    ]
    
    x_padded, y_padded, lengths, pad_mask = collate_fn(batch)
    
    print("x_padded:", x_padded)
    print("y_padded:", y_padded)
    print("lengths:", lengths)
    print("pad_mask:", pad_mask)
    
    # Expected shapes
    print(f"\nx_padded shape: {x_padded.shape}")  # [3, 5]
    print(f"y_padded shape: {y_padded.shape}")    # [3, 5]
    print(f"lengths shape: {lengths.shape}")      # [3]
    print(f"pad_mask shape: {pad_mask.shape}")    # [3, 5]

    # Timing comparison
    import time
    from torch.nn.utils.rnn import pad_sequence
    
    # Generate larger random batch for timing
    num_samples = 1000
    batch_large = [
        (torch.randint(1, 1000, (torch.randint(10, 128, (1,)).item(),)), None)
        for _ in range(num_samples)
    ]
    # Update lengths
    batch_large = [(ids, len(ids)) for ids, _ in batch_large]

    num_runs = 100

    # Time custom collate_fn
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = collate_fn(batch_large)
    custom_time = (time.perf_counter() - start) / num_runs
    print(f"\nCustom collate_fn: {custom_time*1000:.3f} ms")

    # Time custom collate_fn_prealloc
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = collate_fn_prealloc(batch_large)
    prealloc_time = (time.perf_counter() - start) / num_runs
    print(f"Custom collate_fn_prealloc: {prealloc_time*1000:.3f} ms")

    # Time PyTorch's pad_sequence
    def pytorch_collate_fn(batch, pad_token=0, ignore_target=-100):
        tokens = [torch.tensor(ids, dtype=torch.long) for ids, _ in batch]
        targets = [t[1:].clone() for t in tokens]
        lengths = torch.tensor([len(t) for t in tokens], dtype=torch.long)
        
        x_padded = pad_sequence(tokens, batch_first=True, padding_value=pad_token)
        y_padded = pad_sequence(targets, batch_first=True, padding_value=ignore_target)
        pad_mask = (x_padded != pad_token)
        
        return x_padded, y_padded, lengths, pad_mask

    start = time.perf_counter()
    for _ in range(num_runs):
        _ = pytorch_collate_fn(batch_large)
    pytorch_time = (time.perf_counter() - start) / num_runs
    print(f"PyTorch pad_sequence: {pytorch_time*1000:.3f} ms")

    # Try HuggingFace's DataCollatorWithPadding if available
    try:
        from transformers import DataCollatorWithPadding, AutoTokenizer
        
        # HF collator works differently - it expects dict inputs
        def hf_style_collate_fn(batch, pad_token=0, ignore_target=-100):
            tokens = [torch.tensor(ids, dtype=torch.long) for ids, _ in batch]
            max_len = max(len(t) for t in tokens)
            
            x_padded = torch.full((len(batch), max_len), pad_token, dtype=torch.long)
            y_padded = torch.full((len(batch), max_len), ignore_target, dtype=torch.long)
            
            for i, t in enumerate(tokens):
                x_padded[i, :len(t)] = t
                y_padded[i, :len(t)-1] = t[1:]
            
            lengths = torch.tensor([len(t) for t in tokens], dtype=torch.long)
            pad_mask = (x_padded != pad_token)
            
            return x_padded, y_padded, lengths, pad_mask

        start = time.perf_counter()
        for _ in range(num_runs):
            _ = hf_style_collate_fn(batch_large)
        hf_time = (time.perf_counter() - start) / num_runs
        print(f"HF-style (tensor indexing): {hf_time*1000:.3f} ms")
    except ImportError:
        print("HuggingFace transformers not installed, skipping HF-style test")

    print(f"\nSpeedup (PyTorch vs Custom): {custom_time/pytorch_time:.2f}x")
