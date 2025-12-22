import torch

def cross_entropy(logits, targets):
    # logits: [N, C]
    # targets: [N]

    N, C = logits.shape

    max_logit = logits.max(dim=-1)[0].detach().unsqueeze(-1) # [N, 1]

    logits = logits - max_logit # [N, C]

    class_logits = logits[torch.arange(N), targets] # [N]

    loss = -class_logits + torch.log(torch.sum(torch.exp(logits), dim=-1)) # [N]

    return loss.mean()

if __name__ == "__main__":

    N, C = 8, 128

    logits = torch.randn((N, C))
    targets = torch.randint(0, C, size=(N,))

    print(logits.shape)
    print(targets.shape)

    loss_ref = torch.nn.functional.cross_entropy(logits, targets)
    print(loss_ref)

    loss = cross_entropy(logits, targets)
    print(loss)