import torch

d = torch.load("/workspace/lenses/qwen3.5-4b/j-lens/lens.pt", map_location="cpu")

print("keys:", d.keys())
for k, v in d.items():
    if torch.is_tensor(v):
        print(k, "tensor", v.shape, v.dtype)
    else:
        print(k, "=", v)
