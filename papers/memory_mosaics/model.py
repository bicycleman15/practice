## v2 version
def forward(x):
    # x: [B, N, D]

    k = self.W_k(x) # [B, N, D]
    v = self.W_v(x) # [B, N, D]

    # --- Keys: change is here in v2 !!
    g = exp(self.W_g(x)) # [B, N, 1]
    lambda_phi = exp(self.W_lambda(x)) # [B, N, 1]
    # need to use a parallel-scan to speed this up
    k = leaky_average(k, g, lambda_phi) 

    # --- Values: conv over t and t+1 ---
    v[:, :-1] += self.lambda_psi * v[:, 1:]

    # --- Normalize ---
    k = RMSNorm(k)
    v = RMSNorm(v)

    # compute attention / "associative memories"
    y = softmax(k @ k.T) @ v

    o = self.W_o(y)

    return o