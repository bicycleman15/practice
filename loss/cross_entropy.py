import torch

def cross_entropy(logits, targets):
    # logits: [N, C]
    # targets: [N]

    N, C = logits.shape

    max_logit = logits.max(dim=-1)[0].detach().unsqueeze(-1) # [N, 1]

    logits = logits - max_logit # [N, C]

    class_logits = logits[torch.arange(N, device=logits.device), targets] # [N]

    loss = -class_logits + torch.log(torch.sum(torch.exp(logits), dim=-1)) # [N]

    return loss.mean()

# let's do the chunking logic in cross entropy
def cross_entropy_chunked(logits, targets, chunk_size=128):
    # logits: [N, C]
    # targets: [N]

    N, C = logits.shape

    max_logit = logits.max(dim=-1)[0].detach() # [N]

    class_logits = logits[torch.arange(N, device=logits.device), targets] - max_logit # [N]

    denominator = torch.zeros_like(class_logits)

    # now we need to calculate the denominator but in chunks
    # so that we never init the [N, C] another tensor
    # right now I am assuming I am chunking over classes
    i = 0
    while i < C:
        # this doesn't save memory since we instatiate torch.exp()
        # this takes up [N, chunk]
        logit_chunk = logits[:, i : i + chunk_size] # [N, chunk]
        denominator += torch.sum(torch.exp(logit_chunk - max_logit.unsqueeze(-1)), dim=1) # [N]
        i += chunk_size

    loss = -class_logits + torch.log(denominator) # [N]

    return loss.mean()