import torch
m = torch.load("Secondstage_3epoch79.pth")
for k, v in m.items():
	print(k, v.shape)
