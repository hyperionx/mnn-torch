import torch
import torch.nn as nn
import snntorch as snn
import torch.nn.functional as F
from mnn_torch.layers import MemristiveLinearLayer, MemristiveConv2d, HomeostasisDropout


class LayerBuilder:
    """Helper class to build layers based on the configuration."""

    @staticmethod
    def build_fc_layer(in_features, out_features, memristive_config):
        """Builds either a regular Linear layer or MemristiveLinearLayer."""
        if memristive_config.get("ideal", False):
            return nn.Linear(in_features, out_features)
        return MemristiveLinearLayer(in_features, out_features, memristive_config)

    @staticmethod
    def build_conv_layer(
        in_channels, out_channels, kernel_size, stride, padding, memristive_config
    ):
        """Builds either a regular Conv2d layer or MemristiveConv2d."""
        if memristive_config.get("ideal", False):
            return nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            )
        return MemristiveConv2d(
            in_channels, out_channels, kernel_size, stride, padding, memristive_config
        )


class BaseSNN(nn.Module):
    """Base class to handle the basic operations of an SNN."""

    def __init__(self, beta, spike_grad, num_steps, memristive_config):
        super().__init__()
        self.beta = beta
        self.spike_grad = spike_grad
        self.num_steps = num_steps
        self.memristive_config = memristive_config
        self.homeostasis_threshold = memristive_config.get("homeostasis_threshold", 100)

    def initialize_lif(self, num_layers):
        """Initialize the LIF neurons."""
        return [
            snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)
            for _ in range(num_layers)
        ]


class MSNN(BaseSNN):
    def __init__(
        self, num_inputs, num_hidden, num_outputs, num_steps, beta, memristive_config
    ):
        super().__init__(
            beta,
            spike_grad=None,
            num_steps=num_steps,
            memristive_config=memristive_config,
        )

        # Build layers
        self.fc1 = LayerBuilder.build_fc_layer(
            num_inputs, num_hidden, memristive_config
        )
        self.fc2 = LayerBuilder.build_fc_layer(
            num_hidden, num_outputs, memristive_config
        )

        # Initialize LIF neurons
        self.lif1, self.lif2 = self.initialize_lif(2)

        # Initialize homeostasis dropout if needed
        self.drop_layer = (
            HomeostasisDropout()
            if memristive_config.get("homeostasis_dropout", False)
            else None
        )

    def forward(self, x):
        mem1, mem2 = self.lif1.init_leaky(), self.lif2.init_leaky()
        spk1_rec, mem1_rec = [], []
        spk2_rec, mem2_rec = [], []

        for _ in range(self.num_steps):
            # First layer
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            # Collect spikes
            spk1_rec.append(spk1)
            mem1_rec.append(mem1)

            # Apply drop layer
            if self.drop_layer and len(spk1_rec) >= self.homeostasis_threshold:
                spk1_window = torch.stack(
                    spk1_rec[-self.homeostasis_threshold :], dim=0
                )
                spk1 = self.drop_layer(spk1_window)

            # Second layer
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            # Collect spikes
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


class MCSNN(BaseSNN):
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
        super().__init__(beta, spike_grad, num_steps, memristive_config)

        # Store configurations
        self.batch_size = batch_size
        self.stride = stride
        self.num_conv1 = num_conv1
        self.num_conv2 = num_conv2
        self.max_pooling = max_pooling

        # Build convolutional layers
        self.conv1 = LayerBuilder.build_conv_layer(
            input_shape[0], num_conv1, num_kernels, stride, padding, memristive_config
        )
        self.conv2 = LayerBuilder.build_conv_layer(
            num_conv1, num_conv2, num_kernels, stride, padding, memristive_config
        )

        # Initialize LIF neurons
        self.lif1, self.lif2, self.lif3 = self.initialize_lif(3)

        # Calculate flattened size after convolutions and pooling
        self.num_hidden = self._calculate_hidden_size(
            input_shape, num_kernels, stride, padding, max_pooling
        )

        # Build fully connected layer
        self.fc1 = LayerBuilder.build_fc_layer(
            self.num_hidden, num_outputs, memristive_config
        )

        # Initialize HomeostasisDropout layers for each LIF layer before the output
        if self.memristive_config.get("homeostasis_dropout", False):
            self.homeostasis_dropout1 = HomeostasisDropout()
            self.homeostasis_dropout2 = HomeostasisDropout()
        else:
            self.homeostasis_dropout1 = None
            self.homeostasis_dropout2 = None

    def _calculate_hidden_size(
        self, input_shape, kernel_size, stride, padding, max_pooling
    ):
        channels, height, width = input_shape
        # Convolutional layer size calculation (with max pooling) for two convolutional layers
        height = ((height + 2 * padding - kernel_size) // stride + 1) // max_pooling
        width = ((width + 2 * padding - kernel_size) // stride + 1) // max_pooling
        height = ((height + 2 * padding - kernel_size) // stride + 1) // max_pooling
        width = ((width + 2 * padding - kernel_size) // stride + 1) // max_pooling
        return height * width * self.num_conv2

    def forward(self, x):
        mem1, mem2, mem3 = (
            self.lif1.init_leaky(),
            self.lif2.init_leaky(),
            self.lif3.init_leaky(),
        )
        spk1_rec, mem1_rec = [], []
        spk2_rec, mem2_rec = [], []
        spk3_rec, mem3_rec = [], []

        for _ in range(self.num_steps):
            # First convolutional layer + LIF
            cur1 = F.max_pool2d(self.conv1(x), self.max_pooling)
            spk1, mem1 = self.lif1(cur1, mem1)

            # Apply homeostasis dropout after first LIF
            if (
                self.homeostasis_dropout1
                and len(spk1_rec) >= self.homeostasis_threshold
            ):
                spk1_window = torch.stack(
                    spk1_rec[-self.homeostasis_threshold :], dim=0
                )
                spk1 = self.homeostasis_dropout1(spk1_window)

            spk1_rec.append(spk1)
            mem1_rec.append(mem1)

            # Second convolutional layer + LIF
            cur2 = F.max_pool2d(self.conv2(spk1), self.max_pooling)
            spk2, mem2 = self.lif2(cur2, mem2)

            # Apply homeostasis dropout after second LIF
            if (
                self.homeostasis_dropout2
                and len(spk2_rec) >= self.homeostasis_threshold
            ):
                spk2_window = torch.stack(
                    spk2_rec[-self.homeostasis_threshold :], dim=0
                )
                spk2 = self.homeostasis_dropout2(spk2_window)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

            # Fully connected layer + LIF
            cur3 = self.fc1(spk2.view(self.batch_size, -1))
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)
