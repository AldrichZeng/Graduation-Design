import numpy as np

import torch
import torch.autograd as autograd

m=torch.nn.Linear(2,3)
print(m.weight.data.cpu().numpy())
weights=m.weight
y=torch.abs(m.weight)
print(y)
x=np.percentile(y.data.cpu().numpy(),50)
print(x)
z=torch.gt(torch.abs(y.data), x).float()
print(z)


