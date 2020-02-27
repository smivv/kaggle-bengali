import torch
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ExponentialLR
from .cosine import CosineAnnealingWarmUpRestarts
from catalyst.contrib.nn.optimizers import Lookahead, RAdam

# from catalyst.contrib.nn.optimizers.radam import RAdam

model = torch.nn.Linear(2, 1)
optimizer = RAdam(model.parameters(), lr=0.005)
lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer,
                                             T_0=200,
                                             T_mult=2,
                                             eta_max=0.005,
                                             eta_min=0.0000001,
                                             T_up=10,
                                             gamma=0.1,
                                             end_at_zero=True
                                             )
lr_scheduler = ExponentialLR(optimizer, gamma=0.93)
lrs = []

for i in range(200):
    lr_scheduler.step(i)
    lrs.append(
        optimizer.param_groups[0]["lr"]
    )

print(f"50: {lrs[50]}, 100: {lrs[100]}")
print(min(lrs), max(lrs))

plt.plot(lrs)
plt.savefig("/home/smirnvla/PycharmProjects/catalyst-classification/plt.png")
