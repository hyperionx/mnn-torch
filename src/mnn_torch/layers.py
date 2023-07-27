import torch
import torch.nn as nn
import numpy as np
import math


class MemristorLinearLayer(nn.Module):
    """Custom memrisitive layer inherited from torch"""

    def __init__(self, device, size_in, size_out, G_off, G_on, k_V=0.5, ideal=True):
        super().__init__()
        self.device = device
        self.size_in, self.size_out = size_in, size_out
        self.G_off, self.G_on, self.k_V = G_off, G_on, k_V
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)
        self.ideal = ideal

        # initialize weights and biases
        stdv = 1 / np.sqrt(self.size_in)
        nn.init.normal_(self.weights, mean=0, std=stdv)
        nn.init.constant_(self.bias, 0)


    def forward(self, x):
        if self.ideal:
            w_times_x = torch.mm(x, self.weights.t())
            return torch.add(w_times_x, self.bias)  # w times x + b

        inputs = torch.cat([x, torch.ones([x.shape[0], 1]).to(self.device)], 1)
        bias = torch.unsqueeze(self.bias, 0)

        weights_and_bias = torch.cat([self.weights.t(), bias], 0)
        self.outputs = self.memristive_outputs(inputs, weights_and_bias)

        return self.outputs

    def memristive_outputs(self, x, weights_and_bias):
        # input to voltage
        V = self.k_V * x

        max_wb = torch.max(torch.abs(weights_and_bias))

        # Conductance scaling factor
        k_G = (self.G_on - self.G_off) / max_wb
        G_eff = k_G * weights_and_bias

        # Map weights onto conductances.
        G_pos = torch.max(G_eff, torch.Tensor([0]).to(self.device)) + self.G_off
        G_neg = -torch.min(G_eff, torch.Tensor([0]).to(self.device)) + self.G_off

        G = torch.reshape(
            torch.cat((G_pos[:, :, None], G_neg[:, :, None]), -1),
            [G_pos.size(dim=0), -1],
        )

        I = torch.tensordot(V, G, dims=1)
        I_total = I[:, 0::2] - I[:, 1::2]

        k_I = self.k_V * k_G
        y = I_total / k_I

        return y
