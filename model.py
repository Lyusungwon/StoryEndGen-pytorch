import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_


class IEMSAModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, batch):
        pass
        return logit
