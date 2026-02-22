import torch
ckpt = torch.load("vg_objectdetector_pretrained.pth")
print(list(ckpt.keys()))
print(list(ckpt['model'].keys())[:50])