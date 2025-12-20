import torch

def _top_k(logits, k):
    # logits: [B, V]
    top_k_values, top_k_indices = torch.topk(logits, k=k, dim=-1)
    logits = torch.full_like(logits, float("-inf")).scatter_(dim=-1, index=top_k_indices, src=top_k_values) # [B, V]
    return logits

def _top_p(logits, p):
    # logits: [B, V]

    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True) # [B, V]

    sorted_probsum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1) # [B, V]
    mask = (sorted_probsum <= p) # [B, V]

    # make logits -infinity that we don't want to keep
    mask = ~mask # [B, V]

    # ensure we keep atleast one token
    mask[:, 0] = False # don't make the largest logit -inf
    
    sorted_logits = torch.masked_fill(sorted_logits, mask, -float("inf"))

    logits = torch.scatter(logits, dim=1, index=sorted_indices, src=sorted_logits)

    return logits

def sample(logits, temperature, top_k, top_p):
    # assume logits: [B, V]
    # only get a single token logit

    if temperature == 0:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature

    if top_k is not None:
        logits = _top_k(logits, top_k)

    if top_p > 0:
        logits = _top_p(logits, top_p)

    probs = torch.softmax(logits, dim=-1) # [B, V]

    # sample a token now
    return torch.multinomial(probs, num_samples=1).squeeze(-1) # [B, 1]