import torch

q

# KV-cache
k
v

# everything below in pytorch land
s = q @ k.T # load k and "save s"
p = torch.softmax(s, dim=-1) # "load s and save p"
y = p @ v # "load s" and v, and save y finaly

return y
