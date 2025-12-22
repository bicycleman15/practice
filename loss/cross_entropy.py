import torch

def cross_entropy(logits, targets, ignore_index=-100):
    # logits: [N, C]
    # targets: [N]

    N, C = logits.shape

    max_logit = logits.max(dim=-1)[0].detach().unsqueeze(-1) # [N, 1]

    logits = logits - max_logit # [N, C]

    # clamp targets to valid range to avoid indexing errors (ignored indices will be masked out later)
    safe_targets = targets.clone()
    safe_targets[targets == ignore_index] = 0
    
    class_logits = logits[torch.arange(N, device=logits.device), safe_targets] # [N]

    loss = -class_logits + torch.log(torch.sum(torch.exp(logits), dim=-1)) # [N]

    mask = (targets == ignore_index)
    loss[mask] = 0 # don't let autograd handle this

    return loss.sum() / (~mask).sum()

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


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Test 1: Basic test without ignore_index
    print("Test 1: Basic cross entropy (no ignore_index)")
    logits = torch.randn(8, 10, requires_grad=True)
    targets = torch.randint(0, 10, (8,))
    
    my_loss = cross_entropy(logits, targets)
    torch_loss = torch.nn.functional.cross_entropy(logits, targets)
    
    print(f"  My loss: {my_loss.item():.6f}")
    print(f"  Torch loss: {torch_loss.item():.6f}")
    print(f"  Match: {torch.allclose(my_loss, torch_loss)}")
    
    # Test 2: With ignore_index
    print("\nTest 2: Cross entropy with ignore_index")
    logits2 = torch.randn(8, 10, requires_grad=True)
    targets2 = torch.tensor([0, 1, -100, 3, -100, 5, 6, 7])  # indices 2 and 4 should be ignored
    
    my_loss2 = cross_entropy(logits2, targets2, ignore_index=-100)
    torch_loss2 = torch.nn.functional.cross_entropy(logits2, targets2, ignore_index=-100)
    
    print(f"  My loss: {my_loss2.item():.6f}")
    print(f"  Torch loss: {torch_loss2.item():.6f}")
    print(f"  Match: {torch.allclose(my_loss2, torch_loss2)}")
    
    # Test 3: Gradient check with ignore_index
    print("\nTest 3: Gradient check with ignore_index")
    logits3 = torch.randn(8, 10, requires_grad=True)
    logits3_clone = logits3.clone().detach().requires_grad_(True)
    targets3 = torch.tensor([0, -100, 2, -100, 4, 5, -100, 7])
    
    my_loss3 = cross_entropy(logits3, targets3, ignore_index=-100)
    torch_loss3 = torch.nn.functional.cross_entropy(logits3_clone, targets3, ignore_index=-100)
    
    my_loss3.backward()
    torch_loss3.backward()
    
    grad_match = torch.allclose(logits3.grad, logits3_clone.grad, atol=1e-5)
    print(f"  Gradients match: {grad_match}")
    
    # Test 4: All tokens ignored (edge case)
    print("\nTest 4: Edge case - all tokens ignored")
    logits4 = torch.randn(4, 10, requires_grad=True)
    targets4 = torch.tensor([-100, -100, -100, -100])
    
    my_loss4 = cross_entropy(logits4, targets4, ignore_index=-100)
    print(f"  Loss when all ignored: {my_loss4.item()} (expected: nan)")
    
    # Test 5: Custom ignore_index value
    print("\nTest 5: Custom ignore_index value (-1)")
    logits5 = torch.randn(8, 10, requires_grad=True)
    targets5 = torch.tensor([0, 1, -1, 3, -1, 5, 6, 7])
    
    my_loss5 = cross_entropy(logits5, targets5, ignore_index=-1)
    torch_loss5 = torch.nn.functional.cross_entropy(logits5, targets5, ignore_index=-1)
    
    print(f"  My loss: {my_loss5.item():.6f}")
    print(f"  Torch loss: {torch_loss5.item():.6f}")
    print(f"  Match: {torch.allclose(my_loss5, torch_loss5)}")
    
    print("\n✅ All tests completed!")