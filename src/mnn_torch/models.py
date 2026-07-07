import torch
import torch.nn as nn
import snntorch as snn
import torch.nn.functional as F
from mnn_torch.layers import (
    MemristiveLinearLayer,
    MemristiveConv2d,
    HomeostasisDropout,
    HomeostaticRegulariser,
)


def beta_from_tau_leak(tau_leak, dt):
    r"""Map a measured device retention ``tau_leak`` to the LIF membrane decay ``beta``.

    A leaky integrator with time constant :math:`\tau` decays over one timestep of
    length :math:`dt` by the factor :math:`\beta = \exp(-dt/\tau)`. This is the SAME
    exponentially-leaky state, and the SAME :math:`\tau_{\mathrm{leak}}` (trap-discharge
    time, :math:`R_{\mathrm{leak}}C`), that Chapter 5/7 identifies as the eligibility
    trace of reward-modulated learning. Using it here makes the neuron's temporal
    membrane the device's own relaxation: one measured material timescale instantiates
    BOTH the RL eligibility trace and the spiking-classifier's temporal integrator --
    the leaky-integrator primitive the two share. ``beta`` is clamped to ``[0, 1)`` as
    snnTorch requires.
    """
    import math
    beta = math.exp(-float(dt) / float(tau_leak))
    return min(max(beta, 0.0), 1.0 - 1e-6)


class DeviceLeaky(snn.Leaky):
    """A LIF neuron whose membrane leak is set by the MEASURED device retention
    ``tau_leak`` rather than a free hyperparameter: ``beta = exp(-dt/tau_leak)``
    (see :func:`beta_from_tau_leak`). Semantically identical to ``snn.Leaky`` -- it
    only fixes where ``beta`` comes from -- so it is a drop-in whose membrane dynamics
    are device-grounded, matching the eligibility-trace substrate of the RL chapter."""

    def __init__(self, tau_leak, dt, spike_grad=None, **kwargs):
        super().__init__(beta=beta_from_tau_leak(tau_leak, dt),
                         spike_grad=spike_grad, **kwargs)
        self.tau_leak = float(tau_leak)
        self.dt = float(dt)


class AdaptiveLeaky(nn.Module):
    r"""Adaptive leaky integrate-and-fire (ALIF) neuron -- the LSNN working-memory cell
    of Bellec et al. (NeurIPS 2018 / Nat. Commun. 2020).

    A plain LIF neuron has one state, the membrane, and a FIXED threshold. The ALIF
    neuron adds a SECOND, slow per-neuron state: an adaptive threshold. Each spike
    raises the threshold; it then decays with its own long time constant
    :math:`\tau_a \gg \tau_m`:

    .. math::
        v_{t}   &= \beta\, v_{t-1} + I_t - z_{t-1}\,\vartheta_{t-1} \\
        b_{t}   &= \rho\, b_{t-1} + z_{t-1},\quad \rho = e^{-\Delta t/\tau_a} \\
        \vartheta_t &= v_{\mathrm{th}} + \gamma\, b_t \\
        z_t     &= H(v_t - \vartheta_t)

    Information written into the slow adaptation state :math:`b` persists for
    :math:`\sim\tau_a`, which is what gives the LSNN its ``long short-term memory''.
    This is the ENGINEERED-variable route to a slow timescale, and the baseline against
    which :class:`DeviceLeaky` is compared: the device supplies the same slow timescale
    from the material leak alone (``beta = exp(-dt/tau_leak)``), without this extra
    per-neuron state. Uses the fast-sigmoid surrogate gradient by default, matching the
    rest of the package.

    Call convention mirrors ``snn.Leaky`` but carries the extra adaptation state:
    ``spk, mem, b = neuron(input, mem, b)``; ``mem, b = neuron.init_alif(shape)``.
    """

    def __init__(self, beta, tau_a, dt=1.0, gamma=1.6, threshold=1.0, spike_grad=None):
        super().__init__()
        import math
        self.beta = float(beta)
        self.rho = math.exp(-float(dt) / float(tau_a))
        self.tau_a = float(tau_a)
        self.dt = float(dt)
        self.gamma = float(gamma)
        self.threshold = float(threshold)
        if spike_grad is None:
            from snntorch import surrogate
            spike_grad = surrogate.fast_sigmoid(slope=25)
        self.spike_grad = spike_grad

    @staticmethod
    def init_alif(shape):
        """Zero membrane + zero adaptation state of the given ``shape``."""
        return torch.zeros(*shape), torch.zeros(*shape)

    def forward(self, input_, mem, b):
        mem = self.beta * mem + input_                   # leak + integrate this step
        thr = self.threshold + self.gamma * b            # adaptive threshold (prev b)
        spk = self.spike_grad(mem - thr)                 # surrogate-gradient spike
        mem = mem - spk * thr                            # subtract-reset
        b = self.rho * b + spk                           # slow adaptation update
        return spk, mem, b


def _build_homeostasis(memristive_config):
    """Construct the homeostasis module selected by config.

    Returns ``None`` when homeostasis is disabled. By default the
    firing-rate-dependent :class:`HomeostaticRegulariser` (manuscript Eq. homeo)
    is used; setting ``homeostasis_mode='legacy'`` selects the binary
    :class:`HomeostasisDropout` retained for backward compatibility.
    """
    if not memristive_config.get("homeostasis_dropout", False):
        return None
    if memristive_config.get("homeostasis_mode", "regulariser") == "legacy":
        return HomeostasisDropout()
    return HomeostaticRegulariser(
        r_th=memristive_config.get("homeostasis_r_th", 0.8),
        n=memristive_config.get("homeostasis_n", 1.0),
    )


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
        """Initialize the LIF neurons.

        If the config supplies ``lif_tau_leak`` (a measured device retention) the
        neurons are :class:`DeviceLeaky` -- their membrane decay is the device's own
        trap-discharge relaxation (``beta = exp(-dt/tau_leak)``), the same primitive
        as the RL eligibility trace. Otherwise a plain ``snn.Leaky`` with the given
        ``beta`` is used (unchanged behaviour for image tasks)."""
        tau_leak = self.memristive_config.get("lif_tau_leak")
        if tau_leak is not None:
            dt = self.memristive_config.get("lif_dt", 1.0)
            return [
                DeviceLeaky(tau_leak=tau_leak, dt=dt, spike_grad=self.spike_grad)
                for _ in range(num_layers)
            ]
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

        # Initialize homeostasis module if needed (firing-rate-dependent
        # regulariser by default; legacy binary dropout if configured).
        self.drop_layer = _build_homeostasis(memristive_config)

    def forward(self, x):
        mem1, mem2 = self.lif1.init_leaky(), self.lif2.init_leaky()
        spk1_rec, mem1_rec = [], []
        spk2_rec, mem2_rec = [], []

        # The input is constant current-injection: x does not change across timesteps,
        # and the device-fixed fault mask + frozen PF residual are sampled once and cached
        # (see MemristiveLinearLayer). So fc1(x) is loop-invariant -- computing it ONCE
        # instead of num_steps times is numerically identical and removes the dominant
        # redundant cost (the big input-layer Poole-Frenkel per-synapse tensor, evaluated
        # ~num_steps fewer times). fc2 stays in the loop: it depends on spk1, which varies.
        cur1 = self.fc1(x)

        for _ in range(self.num_steps):
            # First layer (current precomputed above; LIF state still steps per timestep)
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


class TemporalMSNN(BaseSNN):
    """Fully-connected spiking network that STREAMS a time-series input across its
    timesteps, so the LIF neurons perform genuine temporal integration of the signal
    as it unfolds -- unlike MSNN, which feeds a static (flattened) input identically
    at every timestep. Built for time-series such as EEG: at network-timestep ``t`` the
    layer sees the ``t``-th sample of the input (the per-channel vector at that time),
    so spike timing and membrane dynamics encode the signal's temporal/spectral
    structure. This is the SNN capability a flattened static encoding discards, and it
    ties to the thesis's retention theme: the device's slow relaxation integrates the
    streamed signal over time.

    Input ``x`` has shape ``(T, batch, num_inputs)`` where T is the number of input
    time samples; the network runs for ``T`` steps (``num_steps`` is ignored / set to
    T). ``num_inputs`` is the per-timestep feature count (e.g. EEG channels)."""

    def __init__(self, num_inputs, num_hidden, num_outputs, beta, memristive_config):
        super().__init__(
            beta, spike_grad=None, num_steps=0, memristive_config=memristive_config
        )
        self.fc1 = LayerBuilder.build_fc_layer(num_inputs, num_hidden, memristive_config)
        self.fc2 = LayerBuilder.build_fc_layer(num_hidden, num_outputs, memristive_config)
        self.lif1, self.lif2 = self.initialize_lif(2)
        self.drop_layer = _build_homeostasis(memristive_config)

    def forward(self, x):
        # x: (T, batch, num_inputs) -- a streamed time series.
        T = x.shape[0]
        mem1, mem2 = self.lif1.init_leaky(), self.lif2.init_leaky()
        spk1_rec, spk2_rec, mem2_rec = [], [], []

        for t in range(T):
            # fc1 runs on THIS timestep's input slice (not a static vector) -- the LIF
            # state carries history across t, integrating the signal over time.
            cur1 = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_rec.append(spk1)

            if self.drop_layer and len(spk1_rec) >= self.homeostasis_threshold:
                spk1_window = torch.stack(spk1_rec[-self.homeostasis_threshold:], dim=0)
                spk1 = self.drop_layer(spk1_window)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
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

        # Build convolutional layers. With ``conv_ideal=True`` the conv layers are
        # standard (ideal) and only the final dense layer carries the Poole-Frenkel
        # device model -- the Joksas et al. methodology (PF on dense layers only),
        # which is fast and avoids the conv PF cost entirely. Homeostasis still acts on
        # the inter-layer spike rates regardless. Default (False) applies PF to the
        # conv layers too (beyond Joksas).
        conv_config = memristive_config
        if memristive_config.get("conv_ideal", False):
            conv_config = {**memristive_config, "ideal": True}
        self.conv1 = LayerBuilder.build_conv_layer(
            input_shape[0], num_conv1, num_kernels, stride, padding, conv_config
        )
        self.conv2 = LayerBuilder.build_conv_layer(
            num_conv1, num_conv2, num_kernels, stride, padding, conv_config
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

        # Initialize homeostasis modules for each LIF layer before the output
        # (firing-rate-dependent regulariser by default; legacy binary dropout
        # if configured).
        self.homeostasis_dropout1 = _build_homeostasis(memristive_config)
        self.homeostasis_dropout2 = _build_homeostasis(memristive_config)

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

            # Fully connected layer + LIF. Use the runtime batch dim (not the
            # hardcoded self.batch_size, which breaks on a final partial batch) and
            # reshape rather than view (the PF conv output may be non-contiguous).
            cur3 = self.fc1(spk2.reshape(spk2.shape[0], -1))
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec), torch.stack(mem3_rec), torch.stack(spk2_rec)


class TemporalMCSNN(MCSNN):
    """Spatiotemporal spiking net: convolution PER FRAME + temporal integration ACROSS
    frames. Combines the two device-grounded pieces -- MemristiveConv2d (spatial features
    per frame, with the binary-spike fast path) and a DeviceLeaky membrane whose leak is
    the measured tau_leak (temporal memory across frames). Where MCSNN feeds a single
    static frame identically at every timestep, this streams a clip: at frame t the conv
    stack runs on x[t], and the LIF membranes carry state across t, so the network
    integrates motion/temporal structure. This is the architecture for spatiotemporal
    data (moving digits, event-camera streams) where BOTH spatial conv and temporal
    memory are needed. Identical construction to MCSNN (same conv dims, hidden-size calc,
    device grounding); only the forward streams over the clip.

    Input ``x`` has shape ``(T, batch, C, H, W)`` -- a clip of T frames. ``num_steps`` is
    ignored; the network runs for T frames."""

    def forward(self, x):
        # x: (T, batch, C, H, W)
        T = x.shape[0]
        mem1, mem2, mem3 = (
            self.lif1.init_leaky(),
            self.lif2.init_leaky(),
            self.lif3.init_leaky(),
        )
        spk1_rec, spk2_rec, spk3_rec, mem3_rec = [], [], [], []

        for t in range(T):
            # conv stack runs on THIS frame; LIF state carries across frames.
            cur1 = F.max_pool2d(self.conv1(x[t]), self.max_pooling)
            spk1, mem1 = self.lif1(cur1, mem1)
            if (self.homeostasis_dropout1
                    and len(spk1_rec) >= self.homeostasis_threshold):
                spk1 = self.homeostasis_dropout1(
                    torch.stack(spk1_rec[-self.homeostasis_threshold:], dim=0))
            spk1_rec.append(spk1)

            cur2 = F.max_pool2d(self.conv2(spk1), self.max_pooling)
            spk2, mem2 = self.lif2(cur2, mem2)
            if (self.homeostasis_dropout2
                    and len(spk2_rec) >= self.homeostasis_threshold):
                spk2 = self.homeostasis_dropout2(
                    torch.stack(spk2_rec[-self.homeostasis_threshold:], dim=0))
            spk2_rec.append(spk2)

            cur3 = self.fc1(spk2.reshape(spk2.shape[0], -1))
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec), torch.stack(mem3_rec), torch.stack(spk2_rec)
