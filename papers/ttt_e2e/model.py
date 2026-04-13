
def init_params():
    s = (2 / (5 * DIM)) ** 0.5
    p = {}
    p["emb"]   = torch.randn(VOCAB, DIM) * 0.02
    for i in range(N_BLOCKS):
        p[f"{i}.nw"] = torch.ones(DIM)
        p[f"{i}.nb"] = torch.zeros(DIM)
        p[f"{i}.w1"] = torch.randn(HIDDEN, DIM) * s
        p[f"{i}.b1"] = torch.zeros(HIDDEN)
        p[f"{i}.w2"] = torch.randn(DIM, HIDDEN) * s
        p[f"{i}.b2"] = torch.zeros(DIM)
    p["fnw"]  = torch.ones(DIM)
    p["fnb"]  = torch.zeros(DIM)
    p["head"] = torch.randn(VOCAB, DIM) * 0.02
    for v in p.values():
        v.requires_grad_(True)
    return p

# there is no model class
# you just handle these param dicts
def fwd(tok_idx, p):
    h = p["emb"][tok_idx].unsqueeze(0)
    for i in range(N_BLOCKS):
        r = h
        h = F.layer_norm(h, (DIM,), p[f"{i}.nw"], p[f"{i}.nb"])
        h = F.gelu(F.linear(h, p[f"{i}.w1"], p[f"{i}.b1"]))
        h = F.linear(h, p[f"{i}.w2"], p[f"{i}.b2"])
        h = r + h
    h = F.layer_norm(h, (DIM,), p["fnw"], p["fnb"])
    return F.linear(h, p["head"])

def ttt_e2e_train_loss(params, tokens):
    T = len(tokens) - 1
    w      = {k: params[k] for k in TTT_KEYS}
    frozen = {k: params[k] for k in FROZEN_KEYS}
 
    total_loss = torch.tensor(0.0, device=DEVICE)
    for t in range(T):
        cur = {**frozen, **w}
        logits = fwd(tokens[t], cur)
        ell_t  = F.cross_entropy(logits, tokens[t + 1].unsqueeze(0))
        total_loss = total_loss + ell_t
 
        grads = torch.autograd.grad(
            ell_t, list(w.values()),
            create_graph=True,       # keep backward ops under autograd
        )

        # replace params
        w = {k: w[k] - INNER_LR * g for k, g in zip(TTT_KEYS, grads)}
 
    return total_loss / T


for step in range(TRAIN_STEPS):
    tokens = sample_seq(SEQ_LEN + 1)

    opt.zero_grad()

    loss = ttt_e2e_train_loss(params, tokens)

    loss.backward()
    opt.step()





params = init_params() # this is a dict

def forward(x: Tensor, params: dict[str, Tensor]):
    # define using torch.nn.functionals only!


# everything here is under autograd!!
def train_step(tokens: Tensor, params: dict[str, Tensor], forward: callable)

    loss = 0
    
    for i in range(len(tokens)):

        # input
        input_id = tokens[i]

        # need to define some target
        # for now, lets choose next token
        target_id = tokens[i+1]

        logits = forward(input_id, params)

        cur_loss = cross_entropy(logits, target_id)

        grad = compute_gradients(cur_loss, params) # grad is the same shape dict as params

        # define a new params dict
        new_params = {}
        for k, v in grad.items():
            # we use GD here
            new_params[k] = params - lr * v # this is under autograd too!

        params = new_params # change reference

        loss += cur_loss # accumulate loss, we need to minimize the sum

    return loss


@torch.no_grad()
def generate(
    prompt: Tensor, params: dict[str, Tensor], forward: callable, max_new_tokens: int
):

    # Phase 1: adapt params on the context first
    for i in range(len(prompt) - 1):
        input_id = prompt[i]
        target_id = prompt[i+1]

        logits = forward(input_id, params)
        cur_loss = cross_entropy(logits, target_id)

        grad = compute_gradients(cur_loss, params)

        new_params = {}
        for k, v in grad.items():
            new_params[k] = params[k] - lr * v
        params = new_params

    # Phase 2: generation using the adapted params and then adapt to your own generations
    tok = prompt[-1]
    generated = []

    for _ in range(max_new_tokens):
        logits = forward(tok, params)
        next_tok = logits.argmax(dim=-1)
        generated.append(next_tok)

        # adapt to your own generations
        # which is okay!
        # this results in these update rules I was talking about
        # compression of long-context
        cur_loss = cross_entropy(logits, next_tok)
        grad = compute_gradients(cur_loss, params)
        new_params = {}
        for k, v in grad.items():
            new_params[k] = params[k] - lr * v
        params = new_params

        tok = next_tok

    return generated


for _ in range(num_steps_in_training):

    tokens = sample_batch()

    loss = train_step(tokens, params, forward)

    grad = compute_gradients(loss, params)

    # now use whatever optimizer you want
    params = optimizer_step(grad, params)

