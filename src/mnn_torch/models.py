import torch
import torch.nn as nn
import snntorch as snn
import torch.nn.functional as F

from mnn_torch.layers import MemristiveLinearLayer, MemristiveConv2d, HomeostasisDropout


class MSNN(nn.Module):
    def __init__(
        self, num_inputs, num_hidden, num_outputs, num_steps, beta, memristive_config
    ):
        super().__init__()
        self.num_steps = num_steps
        self.beta = beta
        self.memristive_config = memristive_config
        self.homeostasis_threshold = memristive_config.get("homeostasis_threshold", 100)

        # Decide which linear layer to use based on the "ideal" key in memristive_config
        self.fc1, self.lif1 = self._build_layer(num_inputs, num_hidden)
        self.fc2, self.lif2 = self._build_layer(num_hidden, num_outputs)

        # Add the custom drop layer if "homeostasis_dropout" is in the config
        if self.memristive_config.get("homeostasis_dropout", False):
            self.drop_layer = HomeostasisDropout()
        else:
            self.drop_layer = None

    def _build_layer(self, in_features, out_features):
        """Helper method to build a linear layer followed by a LIF neuron.
        If 'ideal' is True, use nn.Linear. Otherwise, use MemristiveLinearLayer."""

        if self.memristive_config.get("ideal", False):
            fc = nn.Linear(in_features, out_features)
        else:
            fc = MemristiveLinearLayer(
                in_features, out_features, self.memristive_config
            )
        lif = snn.Leaky(beta=self.beta)
        return fc, lif

    def forward(self, x):
        # Initialize hidden states for both LIF neurons
        mem1, mem2 = self.lif1.init_leaky(), self.lif2.init_leaky()
        spk1_rec, mem1_rec = [], []
        spk2_rec, mem2_rec = [], []

        for _ in range(self.num_steps):
            # Propagate through the first layer and LIF
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            # Collect spike outputs
            spk1_rec.append(spk1)
            mem1_rec.append(mem1)

            # Apply the drop layer after the first LIF if spk1_rec length exceeds the threshold
            if (
                self.drop_layer is not None
                and len(spk1_rec) >= self.homeostasis_threshold
            ):
                spk1_window = torch.stack(
                    spk1_rec[-self.homeostasis_threshold :], dim=0
                )
                spk1 = self.drop_layer(spk1_window)

            # Propagate through the second layer and LIF
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            # Collect spike outputs
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


class MCSNN(nn.Module):
    def __init__(
        self,
        beta,
        spike_grad,
        num_steps,
        batch_size,
        num_kernels,
        num_conv1,
        num_conv2,
        max_pooling,
        num_outputs,
        memristive_config,
        input_shape=(1, 28, 28),  # Default to MNIST input size
        stride=1,
        padding=0,
    ):
        super().__init__()

        # Basic configurations
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.stride = stride
        self.num_conv1 = num_conv1
        self.num_conv2 = num_conv2
        self.max_pooling = max_pooling
        self.memristive_config = memristive_config
        self.ideal = memristive_config.get("ideal", False)  # Check for the ideal config

        # Define convolutional layers (either regular or memristive)
        self.conv1 = self._get_conv_layer(
            input_shape[0], num_conv1, num_kernels, stride, padding
        )
        self.conv2 = self._get_conv_layer(
            num_conv1, num_conv2, num_kernels, stride, padding
        )

        # Define Leaky Integrate-and-Fire (LIF) neurons for each layer
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Calculate flattened size after convolutions and pooling
        self.num_hidden = self._calculate_hidden_size(
            input_shape, num_kernels, stride, padding, max_pooling
        )

        # Define the fully connected layer (either regular or memristive)
        self.fc1 = self._get_fc_layer(self.num_hidden, num_outputs)

    def _get_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Helper function to return either a standard Conv2d or MemristiveConv2d.
        """
        if self.ideal:
            return nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            )
        else:
            return MemristiveConv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                self.memristive_config,
            )

    def _get_fc_layer(self, in_features, out_features):
        """
        Helper function to return either a standard Linear layer or a MemristiveLinearLayer.
        """
        if self.ideal:
            return nn.Linear(in_features, out_features)
        else:
            return MemristiveLinearLayer(
                in_features, out_features, self.memristive_config
            )

    def _calculate_hidden_size(
        self, input_shape, kernel_size, stride, padding, max_pooling
    ):
        """
        Calculates the size of the flattened tensor after convolutional and pooling layers.
        """
        channels, height, width = input_shape

        # First convolutional layer
        height = ((height + 2 * padding - kernel_size) // stride + 1) // max_pooling
        width = ((width + 2 * padding - kernel_size) // stride + 1) // max_pooling

        # Second convolutional layer
        height = ((height + 2 * padding - kernel_size) // stride + 1) // max_pooling
        width = ((width + 2 * padding - kernel_size) // stride + 1) // max_pooling

        return height * width * self.num_conv2

    def forward(self, x):
        # Initialize hidden states for LIF neurons
        mem1, mem2, mem3 = (
            self.lif1.init_leaky(),
            self.lif2.init_leaky(),
            self.lif3.init_leaky(),
        )

        # Prepare to collect spikes and membrane potentials
        spk1_rec, mem1_rec = [], []
        spk2_rec, mem2_rec = [], []
        spk3_rec, mem3_rec = [], []

        # Iterate over time steps
        for _ in range(self.num_steps):
            # First convolutional layer + LIF neuron
            cur1 = F.max_pool2d(self.conv1(x), self.max_pooling)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_rec.append(spk1)
            mem1_rec.append(mem1)

            # Second convolutional layer + LIF neuron
            cur2 = F.max_pool2d(self.conv2(spk1), self.max_pooling)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

            # Fully connected layer + LIF neuron
            cur3 = self.fc1(spk2.view(self.batch_size, -1))
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        # Stack spike and membrane potential outputs along the time axis
        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)
