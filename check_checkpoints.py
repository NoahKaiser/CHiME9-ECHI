import torch

checkpoint = torch.load("/no_backups/s1495/experiments/baseline_1/train_ha/checkpoints/epoch000.pt", map_location="cpu")
print(type(checkpoint))
print(checkpoint.keys())

