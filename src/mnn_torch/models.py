import torch
import torch.nn as nn
import snntorch as snn
import torch.nn.functional as F

from mnn_torch.layers import MemristorLinearLayer

# Define a unified SNN model with flexible layer choice
class MSNN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, num_steps, beta, memristive_config):
        super().__init__()
        self.num_steps = num_steps
        self.beta = beta
        self.memristive_config = memristive_config

        # Decide which linear layer to use based on the "ideal" key in memristive_config
        self.fc1, self.lif1 = self._build_layer(num_inputs, num_hidden)
        self.fc2, self.lif2 = self._build_layer(num_hidden, num_outputs)

    def _build_layer(self, in_features, out_features):
        """Helper method to build a linear layer followed by a LIF neuron.
        If 'ideal' is True, use nn.Linear. Otherwise, use MemristorLinearLayer."""
        if self.memristive_config.get("ideal", False):
            fc = nn.Linear(in_features, out_features)
        else:
            fc = MemristorLinearLayer(in_features, out_features, self.memristive_config)
        lif = snn.Leaky(beta=self.beta)
        return fc, lif

    def forward(self, x):
        # Initialize hidden states for both LIF neurons
        mem1, mem2 = self.lif1.init_leaky(), self.lif2.init_leaky()
        spk2_rec, mem2_rec = [], []

        for _ in range(self.num_steps):
            # Propagate through the first layer and LIF
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            # Propagate through the second layer and LIF
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


import torch
import torch.nn as nn
import snntorch as snn
import torch.nn.functional as F
from mnn_torch.layers import MemristorLinearLayer

class MCSNN(nn.Module):
    def __init__(
        self,
        beta,
        spike_grad,
        batch_size,
        num_kernels,
        num_conv1,
        num_conv2,
        max_pooling,
        num_hidden,
        num_outputs,
        memristive_config
    ):
        super().__init__()

        self.batch_size = batch_size
        self.max_pooling = max_pooling

        # Initialize convolutional layers directly
        self.conv1 = nn.Conv2d(1, num_conv1, num_kernels)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(num_conv1, num_conv2, num_kernels)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Initialize fully connected layer and LIF neuron based on configuration
        if memristive_config.get("ideal", False):
            self.fc1 = nn.Linear(num_hidden, num_outputs)
        else:
            self.fc1 = MemristorLinearLayer(num_hidden, num_outputs, memristive_config)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        # Initialize hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # Forward pass through convolutional layers and LIF neurons
        cur1 = F.max_pool2d(self.conv1(x), self.max_pooling)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = F.max_pool2d(self.conv2(spk1), self.max_pooling)
        spk2, mem2 = self.lif2(cur2, mem2)

        # Flatten output from convolutional layers and pass through fully connected layer
        cur3 = self.fc1(spk2.view(self.batch_size, -1))
        spk3, mem3 = self.lif3(cur3, mem3)

        return spk3, mem3
