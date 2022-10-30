import numpy as np
import torch
features=torch.Tensor([[1,2,3],[0,1,1]])
print(features.pow(2).mean())
