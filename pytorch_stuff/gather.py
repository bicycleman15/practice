import torch

init_tensor = [
    [10, 20, 30, 40],
    [400, 50, 60, 500],
    [70, 80, 90, 100]
] # 3x4

x = torch.tensor(init_tensor)

# index = torch.tensor([2, 3])
# y = torch.index_select(input=x, dim=1, index=index)
# print(y)

target = torch.tensor([2, 3, 1]).unsqueeze(0) # [1, 3]
y = torch.gather(input=x, dim=1, index=target)
print(y)