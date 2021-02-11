import torch
import torch.nn as nn


class MeanAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_self, x_neigh):
        if len(x_self.shape) == 2:
            x_self = x_self.view(x_self.shape[0], -1, x_self.shape[1])

        return torch.cat([x_self, x_neigh], axis=1).mean(1).view(x_self.shape[0], 1, x_self.shape[2])
