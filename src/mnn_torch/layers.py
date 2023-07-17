import torch
import torch.nn as nn
import math


class MemristorLinearLayer(nn.Module):
    """Custom memrisitive layer inherited from torch"""

    def __init__(self, size_in, size_out, G_off, G_on, k_V=0.5, ideal=True):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.G_off, self.G_on, self.k_V = G_off, G_on, k_V
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)
        self.ideal = ideal

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))  # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        if self.ideal:
            w_times_x = torch.mm(x, self.weights.t())
            return torch.add(w_times_x, self.bias)  # w times x + b

        ones = torch.ones([x.shape[0], 1])
        inputs = torch.cat([x, ones], 1)

        bias = torch.unsqueeze(self.bias, 0)
        combined_weights = torch.cat([self.weights.t(), bias], 0)
        self.outputs = self.memristive_outputs(
            inputs, combined_weights, self.G_off, self.G_on, self.k_V
        )

        return self.outputs

    def memristive_outputs(self, x, weights):
        # input to voltage
        V = self.k_V * x

        max_weights = torch.max(torch.abs(weights))

        # Conductance scaling factor
        k_G = (self.G_on - self.G_off) / max_weights
        G_eff = k_G * weights

        # Map weights onto conductances.
        G_pos = torch.max(G_eff, torch.Tensor([0])) + self.G_off
        G_neg = -torch.max(G_eff, torch.Tensor([0])) + self.G_off

        G = torch.reshape(
            torch.cat((G_pos[:, :, None], G_neg[:, :, None]), -1),
            [G_pos.size(dim=0), -1],
        )

        I = torch.tensordot(V, G, dims=1)
        I_total = I[:, 0::2] - I[:, 1::2]

        k_I = self.k_V * k_G
        y = I_total / k_I

        return y
