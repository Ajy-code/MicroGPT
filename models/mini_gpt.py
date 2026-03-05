import torch
import torch.nn as nn
from torch.nn import functional as F

with open("../data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Longueur du corpus :", len(text))
print(text[:500])