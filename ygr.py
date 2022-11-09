import numpy as np
import torch

t1 = torch.FloatTensor(7,1143,80)
t2 = torch.FloatTensor(1143,80)
t3=torch.full([t1.shape[0]],t1.shape[1])
print(t3)
# lista=[t1,t2]
# ta = torch.cat(lista, dim=0).reshape(len(lista),-1,80)
# print(ta.shape)
# print(ta)

#torch.Size([2, 1143, 80])

