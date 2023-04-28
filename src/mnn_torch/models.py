import torch
import torch.nn as nn
import snntorch as snn

from mnn_torch.layers import MemristorDenseLayer

class BasicModel(nn.Module):
    """ Basic memrisitive model with spiking neural network """
    def __init__(self):
        super().__init__()
        # Initialize layers 

        self.flatten = nn.Flatten(start_dim=1)
        self.linear = MemristorDenseLayer(784, 10, ideal=False)
        self.leaky = snn.Leaky(beta=0.5)

    def forward(self, x):

        mem = self.leaky.init_leaky()

        spk_rec = []
        mem_rec = []

        for step in range(100):
            f1 = self.flatten(x)
            cur = self.linear(f1)
            spk, mem = self.leaky(cur, mem)
            spk_rec.append(spk)
            mem_rec.append(mem)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)