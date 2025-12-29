"""



"""


import torch
import torch.nn as nn

class SGD:
    def __init__(self, parameters, lr, weight_decay):
        
        # exhaust the generator
        self.parameters = [x for x in parameters]
        self.lr = lr

    @torch.no_grad()
    def step(self):
        for param in self.parameters:
            # subtract with its gradient
            param -= param.grad * self.lr + self.weight_decay * param

    @torch.no_grad()
    def zero_grad(self):
        for param in self.parameters:
                if param.grad is not None:
                    # set grads to zero
                    param.grad.zero_()


if __name__ == "__main__":

    dim = 32

    model = torch.nn.Sequential(
        nn.Linear(32, 32, bias=True),
        nn.LayerNorm(32, eps=1e-5),
        nn.Linear(32, 32, bias=False),
        nn.RMSNorm(32, eps=1e-5)
    )

    print(model)

    optimizer = SGD(model.parameters(), lr=1e-3)

    x = torch.randn(1, 32)

    for i in range(1000):

        optimizer.zero_grad()

        out = model(x)
        loss = out.sum()

        loss.backward()

        optimizer.step()

        if i % 100 == 0:
            print(f"loss at iter {i}:", loss.item())
