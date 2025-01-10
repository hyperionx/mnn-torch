import torch
import torch.nn as nn
import numpy as np

from mnn_torch.effects import disturb_conductance_fixed, disturb_conductance_device


class MemristiveLinearLayer(nn.Module):
    """
    Custom memristive linear layer.
    This layer simulates the behavior of memristors, optionally applying
    disturbances and non-ideal effects based on the provided configuration.
    """

    def __init__(self, size_in, size_out, memristive_config):
        """
        Initializes the memristive linear layer.

        Args:
        - size_in (int): Number of input features.
        - size_out (int): Number of output features.
        - memristive_config (dict): Configuration dictionary containing:
            - 'ideal': If True, the layer behaves like a standard linear layer.
            - 'disturb_conductance': Whether to disturb conductance values.
            - 'disturb_mode': The mode of disturbance ('fixed' or 'device').
            - 'disturbance_probability': The probability for disturbance.
            - 'G_off', 'G_on': Minimum and maximum conductance values.
            - 'R', 'c', 'd_epsilon', 'k_V': Parameters for non-ideal effects.
        """
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out

        # Load configuration
        self.disturb_conductance = memristive_config.get("disturb_conductance", False)
        self.disturb_mode = memristive_config.get("disturb_mode", "fixed")
        self.disturbance_probability = memristive_config.get(
            "disturbance_probability", 0.1
        )

        # Initialize weights and biases
        self.weights = nn.Parameter(torch.Tensor(size_out, size_in))
        self.bias = nn.Parameter(torch.Tensor(size_out))
        stdv = 1 / np.sqrt(self.size_in)
        nn.init.normal_(self.weights, mean=0, std=stdv)
        nn.init.constant_(self.bias, 0)

        self.G_off = memristive_config["G_off"]
        self.G_on = memristive_config["G_on"]
        self.R = memristive_config["R"]
        self.c = memristive_config["c"]
        self.d_epsilon = memristive_config["d_epsilon"]
        self.k_V = memristive_config["k_V"]

    def forward(self, x):
        """
        Forward pass for the memristive linear layer.

        Args:
        - x (Tensor): Input tensor of shape (batch_size, size_in).

        Returns:
        - Tensor: Output tensor of shape (batch_size, size_out).
        """
        device = x.device

        # Include bias in input
        inputs = torch.cat([x, torch.ones([x.shape[0], 1], device=device)], dim=1)
        bias = torch.unsqueeze(self.bias, dim=0).to(device)
        weights_and_bias = torch.cat([self.weights.t(), bias], dim=0)

        # Compute the memristive outputs
        outputs = self.memristive_outputs(inputs, weights_and_bias)
        return outputs

    def memristive_outputs(self, x, weights_and_bias):
        """
        Computes the memristive layer outputs based on voltage, conductance,
        and current relationships.

        Args:
        - x (Tensor): Input tensor with bias included, shape (batch_size, size_in + 1).
        - weights_and_bias (Tensor): Weights and bias tensor, shape (size_in + 1, size_out).

        Returns:
        - Tensor: Output tensor of shape (batch_size, size_out).
        """
        # Voltage calculation
        V = self.k_V * x

        # Conductance scaling factors
        max_wb = torch.max(torch.abs(weights_and_bias))
        k_G = (self.G_on - self.G_off) / max_wb
        k_I = self.k_V * k_G
        G_eff = k_G * weights_and_bias

        # Map weights to positive and negative conductances
        G_pos = self.G_off + torch.maximum(G_eff, torch.tensor(0.0, device=x.device))
        G_neg = self.G_off - torch.minimum(G_eff, torch.tensor(0.0, device=x.device))
        G = torch.cat((G_pos[:, :, None], G_neg[:, :, None]), dim=-1).reshape(
            G_pos.size(0), -1
        )

        # Disturb conductance based on the selected mode
        if self.disturb_conductance:
            if self.disturb_mode == "fixed":
                G = disturb_conductance_fixed(
                    G, self.G_on, true_probability=self.disturbance_probability
                )
            elif self.disturb_mode == "device":
                G = disturb_conductance_device(
                    G,
                    self.G_on,
                    self.G_off,
                    true_probability=self.disturbance_probability,
                )

        # Compute current
        I_ind = torch.unsqueeze(V, dim=-1) * torch.unsqueeze(G, dim=0)
        I = torch.sum(I_ind, dim=1)
        I_total = I[:, 0::2] - I[:, 1::2]

        # Output calculation
        y = I_total / k_I
        return y


class MemristiveConv2d(nn.Module):
    """
    Custom memristive 2D convolutional layer.
    This layer simulates the behavior of memristors, optionally applying
    disturbances and non-ideal effects based on the provided configuration.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        memristive_config=None,
    ):
        """
        Initializes the memristive convolutional layer.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = self._to_tuple(kernel_size)
        self.stride = self._to_tuple(stride)
        self.padding = self._to_tuple(padding)

        # Load configuration
        self.ideal = memristive_config.get("ideal", True)
        self.disturb_conductance = memristive_config.get("disturb_conductance", False)
        self.disturb_mode = memristive_config.get("disturb_mode", "fixed")
        self.disturbance_probability = memristive_config.get(
            "disturbance_probability", 0.1
        )

        # Initialize weights and biases
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        stdv = 1 / np.sqrt(in_channels * np.prod(self.kernel_size))
        nn.init.normal_(self.weight, mean=0, std=stdv)
        nn.init.constant_(self.bias, 0)

        # Load non-ideal parameters from config
        self.G_off = memristive_config["G_off"]
        self.G_on = memristive_config["G_on"]
        self.R = memristive_config["R"]
        self.c = memristive_config["c"]
        self.d_epsilon = memristive_config["d_epsilon"]
        self.k_V = memristive_config["k_V"]

    def _to_tuple(self, value):
        """Ensures kernel size and padding are tuples."""
        return value if isinstance(value, tuple) else (value, value)

    def forward(self, x):
        """
        Forward pass for the memristive convolutional layer.

        Args:
        - x (Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
        - Tensor: Output tensor of shape (batch_size, out_channels, new_height, new_width).
        """
        device = x.device

        # Memristive weight computation
        weights = self.memristive_weights(self.weight)

        # Apply convolution
        output = nn.functional.conv2d(x, weights, self.bias, self.stride, self.padding)
        return output

    def memristive_weights(self, weights):
        """
        Adjusts weights to simulate memristive properties.

        Args:
        - weights (Tensor): Original weights of shape (out_channels, in_channels, kernel_height, kernel_width).

        Returns:
        - Tensor: Adjusted weights with memristive properties.
        """
        if self.ideal:
            return weights

        # Voltage-based scaling (example)
        V = (
            self.k_V * weights
        )  # Shape: (out_channels, in_channels, kernel_height, kernel_width)
        max_w = torch.max(torch.abs(weights))
        k_G = (self.G_on - self.G_off) / max_w
        k_I = self.k_V * k_G
        G_eff = k_G * weights

        # Map weights to conductance (pos and neg parts)
        G_pos = self.G_off + torch.maximum(
            G_eff, torch.tensor(0.0, device=weights.device)
        )
        G_neg = self.G_off - torch.minimum(
            G_eff, torch.tensor(0.0, device=weights.device)
        )

        # Combine G_pos and G_neg into one tensor of the correct shape (out_channels, in_channels, kernel_height, kernel_width)
        G = G_pos - G_neg  # Simply subtracting for now to get a single tensor back

        # Disturb conductance if necessary
        if self.disturb_conductance:
            if self.disturb_mode == "fixed":
                G = disturb_conductance_fixed(
                    G, self.G_on, true_probability=self.disturbance_probability
                )
            elif self.disturb_mode == "device":
                G = disturb_conductance_device(
                    G,
                    self.G_on,
                    self.G_off,
                    true_probability=self.disturbance_probability,
                )

        # Compute the final current and return adjusted weights
        I_total = G * V  # Apply the conductance adjustment
        y = I_total / k_I  # Final adjusted weights
        return y


class HomeostasisDropout(nn.Module):
    def __init__(self):
        """
        A custom layer that drops the spike output if it has been spiking continuously
        over all time steps (across the entire sequence).
        """
        super(HomeostasisDropout, self).__init__()

    def forward(self, spk_rec):
        """
        Forward pass that checks if the spikes are repeated across all time steps
        and sets those to zero if they are continuously '1' over the entire sequence.

        Args:
        - spk_rec: A tensor containing the spikes over time (num_steps, batch_size, num_features).

        Returns:
        - A tensor with the spikes set to zero if they were '1' continuously across the entire sequence.
        """

        # The shape of spk_rec is (num_steps, batch_size, num_features)
        num_steps, batch_size, *feature_dims = spk_rec.shape

        # Check for continuous spikes across all time steps
        # Compare each time step with the next one to find continuous sequences of 1s
        continuous_spikes = (spk_rec[1:] == 1) & (
            spk_rec[:-1] == 1
        )  # Shape: (num_steps-1, batch_size, num_features)

        # We want to drop the spikes that were continuous across all time steps
        # The spike is dropped if it was '1' in all steps
        drop_mask = torch.cat(
            [
                torch.zeros(1, batch_size, *feature_dims, device=spk_rec.device),
                continuous_spikes,
            ],
            dim=0,
        )

        # Drop the spikes that were continuous across all time steps by applying the drop mask
        spk_rec = spk_rec * (1 - drop_mask).float()

        # Return only the latest step after applying the drop mask
        return spk_rec[-1]  # Select the last step in the sequence
