import os
import torch
import pandas as pd

# t = pd.read_csv(os.path.join('cowboyoutfits', 'test.csv'))
#
# print(type(t.id.values))
# print(t.id.values.shape)
# print(type(t.id.values.tolist()))

a = [1, 2]
b = torch.tensor([3, 4])

for i, (c,d) in enumerate(zip(a,b)):
    print(i)
    print(c,d)