import torch
import torch.nn as nn
import numpy as np
from scipy.stats import truncnorm, lognorm

from mnn_torch.effects import disturb_conductance_fixed

class MemristorLinearLayer(nn.Module):
    """Custom memristive layer inherited from torch"""

    def __init__(self, size_in, size_out, memrisitive_config):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.weights = nn.Parameter(torch.Tensor(size_out, size_in))
        self.bias = nn.Parameter(torch.Tensor(size_out))

        self.ideal = memrisitive_config["ideal"]
        self.disturb_conductance = memrisitive_config["disturb_conductance"]

        # Initialize weights and biases
        stdv = 1 / np.sqrt(self.size_in)
        nn.init.normal_(self.weights, mean=0, std=stdv)
        nn.init.constant_(self.bias, 0)

        if not self.ideal:
            # Use precomputed Poole-Frenkel parameters
            self.G_off = memrisitive_config["G_off"]
            self.G_on = memrisitive_config["G_on"]
            self.R = memrisitive_config["R"]
            self.c = memrisitive_config["c"]
            self.d_epsilon = memrisitive_config["d_epsilon"]
            self.k_V = memrisitive_config["k_V"]

    def forward(self, x):
        # Ensure that all tensors are on the same device as the input
        device = x.device

        if self.ideal:
            # Simple weighted sum: w * x + b
            w_times_x = torch.mm(x, self.weights.to(device))
            return torch.add(w_times_x, self.bias.to(device))

        # Include bias in input
        inputs = torch.cat([x, torch.ones([x.shape[0], 1], device=device)], 1)
        bias = torch.unsqueeze(self.bias, 0).to(device)
        weights_and_bias = torch.cat([self.weights.t(), bias], 0)

        # Compute memristive outputs
        outputs = self.memristive_outputs(inputs, weights_and_bias)
        return outputs

    def memristive_outputs(self, x, weights_and_bias):
        # Voltage calculation
        V = self.k_V * x

        # Conductance scaling
        max_wb = torch.max(torch.abs(weights_and_bias))
        k_G = (self.G_on - self.G_off) / max_wb
        k_I = self.k_V * k_G
        G_eff = k_G * weights_and_bias

        # Map weights to conductances
        G_pos = self.G_off + torch.max(G_eff, torch.tensor(0.0))
        G_neg = self.G_off - torch.min(G_eff, torch.tensor(0.0))
        G = torch.cat((G_pos[:, :, None], G_neg[:, :, None]), dim=-1).reshape(
            G_pos.size(0), -1
        )

        if self.disturb_conductance:
            G = disturb_conductance_fixed(G, self.G_on, true_probability=0.5)

        # Compute current
        I_ind = torch.unsqueeze(V, -1) * torch.unsqueeze(G, 0)
        I = torch.sum(I_ind, dim=1)
        I_total = I[:, 0::2] - I[:, 1::2]

        # Output calculation
        y = I_total / k_I
        return y


# class StuckDeviceDropout(nn.Module):
#     def __init__(self, median_range, proportion_stuck=0.1, seed=None):
#         """
#         Custom dropout layer to simulate stuck devices in memristive systems.

#         Args:
#             median_range (float): Median value of the conductance range (G_on - G_off).
#             proportion_stuck (float): Proportion of devices to be stuck.
#             seed (int, optional): Random seed for reproducibility.
#         """
#         super().__init__()
#         self.median_range = median_range
#         self.proportion_stuck = proportion_stuck
#         self.seed = seed
#         if seed is not None:
#             torch.manual_seed(seed)

#     def _generate_stuck_pdf(self, size):
#         """
#         Generate the probability density function for stuck devices using KDE with truncated normal distributions.

#         Args:
#             size (int): Number of devices.

#         Returns:
#             torch.Tensor: Stuck conductance values.
#         """
#         lower, upper = 0, 1
#         mean, std_dev = 0.5, 0.1  # Assuming normalized range for conductance

#         # Generate truncated normal distribution
#         a, b = (lower - mean) / std_dev, (upper - mean) / std_dev
#         stuck_values = truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=size)

#         # Reflect values to mitigate bias from clipping at zero
#         stuck_values = np.abs(stuck_values)
#         return torch.tensor(stuck_values, dtype=torch.float32)

#     def _apply_device_variability(self, values):
#         """
#         Apply device-to-device variability using a lognormal distribution.

#         Args:
#             values (torch.Tensor): Input conductance values.

#         Returns:
#             torch.Tensor: Modified conductance values with variability.
#         """
#         shape = 0.1  # Lognormal shape parameter for variability
#         scale = 1.0
#         variability = lognorm.rvs(shape, scale=scale, size=values.size(0))
#         return values * torch.tensor(variability, dtype=torch.float32)

#     def forward(self, x):
#         """
#         Apply the custom dropout mechanism.

#         Args:
#             x (torch.Tensor): Input tensor.

#         Returns:
#             torch.Tensor: Output tensor with simulated stuck devices.
#         """
#         if not self.training or self.proportion_stuck <= 0:
#             return x

#         # Determine the number of devices to be stuck
#         num_devices = x.numel()
#         num_stuck = int(self.proportion_stuck * num_devices)

#         # Generate stuck values
#         stuck_values = self._generate_stuck_pdf(num_stuck)
#         stuck_values = self._apply_device_variability(stuck_values)

#         # Create a mask for stuck devices
#         stuck_mask = torch.zeros(num_devices, dtype=torch.bool)
#         stuck_indices = torch.randperm(num_devices)[:num_stuck]
#         stuck_mask[stuck_indices] = True

#         # Apply the stuck behavior to the input tensor
#         x_flat = x.flatten()
#         x_flat[stuck_mask] = stuck_values
#         return x_flat.view_as(x)