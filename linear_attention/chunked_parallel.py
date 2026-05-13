import torch


q
k
v

# create buffer for
s = torch.zeros((B, N // chunk_size, d, d))


## kernel begins below
## sequential kernel

# this is on-chip buffer
s_cur = torch.zeros((B, d, d))
start = 0

for i in range(0, N // chunk_size):

    end = i + chunk_size

    # load k, v from HBM
    key = k[:, start:end, :] # [B, C, d]
    value = v[: start:end, :] # [B, C, d]

    s_cur += key.T @ value # [B, d, d] # this is a matmul, tensor cores :)

    # save on HBM
    s[:, i, :, :] = s_cur

    start += chunk_size


## parallel kernel begins here

# for all positions in parallel
# thread gives you the chunk

for i in range(0, N // chunk_size): # this for loop can happen in parallel now!

    start = i * chunk_size
    end = (i + 1) * chunk_size

    # load s_cur, q, k, v from HBM
    query = q[:, start:end, :] # [B, C, d]
    key = k[:, start:end, :] # [B, C, d]
    value = v[:, start:end, :] # [B, C, d]

    # load the s that we checkpointed earlier in HBM
    s_before = s[:, i, :, :] # [B, d, d]

    # 1. compute output from history
    o_across = query @ s_before # [B, C, d] @ [B, d, d] = [B, C, d]

    # 2. compute output in current chunk 
    o_within = (query @ key.T) @ value # [B, C, d]

    # save this output in HBM
    o = o_across + o_within # [B, C, d]

