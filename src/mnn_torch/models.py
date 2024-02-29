import torch
import torch.nn as nn
import snntorch as snn
import torch.nn.functional as F

from mnn_torch.layers import MemristorLinearLayer


# Define Network

class SNN(nn.Module):
    """Basic memrisitive model with spiking neural network"""

    def __init__(
        self,
        device,
        num_inputs,
        num_hidden,
        num_outputs,
        num_steps,
        beta,
    ):
        super().__init__()

        self.num_steps = num_steps

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden, device=device)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs, device=device)
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


    ):
        super().__init__()

        self.num_steps = num_steps

        # Initialize layers
        # self.fc1 = nn.Linear(num_inputs, num_hidden, device=device)
        self.fc1 = MemristorLinearLayer(
            device, num_inputs, num_hidden, memrisitive_config
        )
        self.lif1 = snn.Leaky(beta=beta)
        # self.fc2 = nn.Linear(num_hidden, num_outputs, device=device)
        self.fc2 = MemristorLinearLayer(
            device, num_hidden, num_outputs, memrisitive_config
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

class CSNN(nn.Module):
    def __init__(
        self,
        device,
        beta,
        spike_grad,
        batch_size,
        num_kernels,
        num_conv1,
        num_conv2,
        max_pooling,
        num_hidden,
        num_outputs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.max_pooling = max_pooling

        # Initialize layers
        self.conv1 = nn.Conv2d(1, num_conv1, num_kernels, device=device)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(num_conv1, num_conv2, num_kernels, device=device)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = nn.Linear(num_hidden, num_outputs, device=device)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        cur1 = F.max_pool2d(self.conv1(x), self.max_pooling)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = F.max_pool2d(self.conv2(spk1), self.max_pooling)
        spk2, mem2 = self.lif2(cur2, mem2)

        cur3 = self.fc1(spk2.view(self.batch_size, -1))
        spk3, mem3 = self.lif3(cur3, mem3)

        return spk3, mem3


class MCSNN(nn.Module):
    def __init__(
        self,
        device,
        beta,
        spike_grad,
        batch_size,
        num_kernels,
        num_conv1,
        num_conv2,
        max_pooling,
        num_hidden,
        num_outputs,
        memrisitive_config,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.max_pooling = max_pooling

        # Initialize layers
        self.conv1 = nn.Conv2d(1, num_conv1, num_kernels, device=device)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(num_conv1, num_conv2, num_kernels, device=device)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        # self.fc1 = nn.Linear(num_hidden, num_outputs, device=device)
        self.fc1 = MemristorLinearLayer(
            device, num_hidden, num_outputs, memrisitive_config
        )
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        cur1 = F.max_pool2d(self.conv1(x), self.max_pooling)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = F.max_pool2d(self.conv2(spk1), self.max_pooling)
        spk2, mem2 = self.lif2(cur2, mem2)

        cur3 = self.fc1(spk2.view(self.batch_size, -1))
        spk3, mem3 = self.lif3(cur3, mem3)

        return spk3, mem3
