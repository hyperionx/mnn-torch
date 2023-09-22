import torch
import torch.nn as nn
import snntorch as snn

from mnn_torch.layers import MemristorLinearLayer


# Define Network
class MSNN(nn.Module):
    """Basic memrisitive model with spiking neural network"""

    def __init__(
        self,
        device,
        num_inputs,
        num_hidden,
        num_outputs,
        num_steps,
        beta,
        memrisitive_config,
        ideal=True,
    ):
        super().__init__()

        self.num_steps = num_steps

        # Initialize layers
        self.fc1 = MemristorLinearLayer(
            device, num_inputs, num_hidden, memrisitive_config, ideal=ideal
        )
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = MemristorLinearLayer(
            device, num_hidden, num_outputs, memrisitive_config, ideal=ideal
        )
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for _ in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
