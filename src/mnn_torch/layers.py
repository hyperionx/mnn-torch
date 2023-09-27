import torch
import torch.nn as nn
import numpy as np

from mnn_torch.effects import (
    compute_PooleFrenkel_parameters,
    disturb_conductance,
)


class MemristorLinearLayer(nn.Module):
    """Custom memrisitive layer inherited from torch"""

    def __init__(self, device, size_in, size_out, memrisitive_config):
        super().__init__()
        self.device = device
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        self.ideal = memrisitive_config["ideal"]
        self.disturb_conductance = memrisitive_config["disturb_conductance"]
        self.memrisitive_config = memrisitive_config

        # initialize weights and biases
        stdv = 1 / np.sqrt(self.size_in)
        nn.init.normal_(self.weights, mean=0, std=stdv)
        nn.init.constant_(self.bias, 0)

        if not self.ideal:
            (
                self.G_off,
                self.G_on,
                self.R,
                self.c,
                self.d_epsilon,
            ) = compute_PooleFrenkel_parameters(
                self.memrisitive_config["experimental_data"]
            )
            self.k_V = self.memrisitive_config["k_V"]

    def forward(self, x):
        if self.ideal:  # w times x + b
            w_times_x = torch.mm(x, self.weights.t())
            return torch.add(w_times_x, self.bias)

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
        k_I = self.k_V * k_G
        G_eff = k_G * weights_and_bias

        # Map weights onto conductances.
        G_pos = self.G_off + torch.max(G_eff, torch.Tensor([0]).to(self.device))
        G_neg = self.G_off - torch.min(G_eff, torch.Tensor([0]).to(self.device))

        G = torch.reshape(
            torch.cat((G_pos[:, :, None], G_neg[:, :, None]), -1),
            [G_pos.size(dim=0), -1],
        )

        if self.disturb_conductance:
            G = disturb_conductance(G, self.G_on, true_probability=0.5)

        I_ind = torch.unsqueeze(V, -1) * torch.unsqueeze(G, 0)
        I = torch.sum(I_ind, dim=1)

        I_total = I[:, 0::2] - I[:, 1::2]

        y = I_total / k_I
        return y
