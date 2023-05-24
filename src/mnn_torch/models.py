import torch
import torch.nn as nn
import snntorch as snn

from mnn_torch.layers import MemristorLinearLayer


class BasicModel(nn.Module):
    """ Basic memrisitive model with spiking neural network """

    def __init__(self):
        super().__init__()
        # Initialize layers

        self.flatten = nn.Flatten(start_dim=1)
        self.linear1 = MemristorLinearLayer(784, 25, ideal=False)
        self.leaky1 = snn.Leaky(beta=0.95)
        self.linear2 = MemristorLinearLayer(25, 10, ideal=False)
        self.leaky2 = snn.Leaky(beta=0.95)

    def forward(self, x):

        mem1 = self.leaky1.init_leaky()
        mem2 = self.leaky2.init_leaky()

        spk2_rec = []
        mem2_rec = []

        for step in range(25):
            x = self.flatten(x)
            cur1 = self.linear1(x)
            spk1, mem1 = self.leaky1(cur1, mem1)
            cur2 = self.linear2(spk1)
            spk2, mem2 = self.leaky2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
