import torch
# x = torch.arange(0, 6)
# x = x.unsqueeze(dim=1)
# ones = torch.ones(3, 3)
# # x= (1-torch.tril(ones, diagonal=0)).type(torch.bool)
# x= (torch.triu(ones, diagonal=1)).type(torch.bool)
# print(ones)
# print(ones.shape)

# print(x)
# print(x.shape)
# n_x = 3
# n_y = 5
# x = torch.linspace(0, n_x, n_x+1)  # (n_x,)
# y = torch.linspace(0, n_y, n_y+1)  # (n_y,)
# # Build grid
# X, Y = torch.meshgrid(x, y, indexing='xy')  # both (n_x, n_y)
# z = X*Y
# print(z[torch.tensor([1,3]), torch.tensor([3,3])])

# n_fused = 10
# i_p = 6
# casuality_mask = torch.ones(size=(n_fused, n_fused), dtype=torch.bool) 
# print(casuality_mask)
# casuality_mask = torch.triu(casuality_mask, diagonal=1)
# print(casuality_mask)
# # the inputs are not required to look only to the left, bi-derectiona attntion allowed
# casuality_mask[:i_p, :i_p,] = False
# print(casuality_mask)
n_batch = 6
n_samples = 10
print(-(-n_samples//n_batch))