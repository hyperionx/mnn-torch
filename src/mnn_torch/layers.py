import torch
import torch.nn as nn
import numpy as np

from mnn_torch.effects import (
    disturb_conductance_fixed,
    disturb_conductance_device,
    sample_device_fault,
    apply_device_fault,
    compute_PooleFrenkel_current_torch,
    sample_PooleFrenkel_parameters_torch,
    cholesky_factor_from_covariance,
)


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
        # Lognormal D2D width for the fixed-fault ('device_fixed') protocol.
        self.disturbance_sigma_g = memristive_config.get("disturbance_sigma_g", 0.5)
        # Stuck-fault polarity for 'device_fixed': 'split' (G_off or G_on, symmetric
        # default), 'off' (dead/open synapses) or 'on' (shorted/saturated synapses).
        # Load-bearing for the homeostasis stuck-on vs stuck-off mechanism test.
        self.disturbance_stuck_polarity = memristive_config.get(
            "disturbance_stuck_polarity", "split"
        )
        # Cache for the fixed device-fault realisation (Section 6.3 protocol):
        # sampled once on first forward and reused for the rest of the run.
        self._device_fault = None
        self.homeostasis_dropout = memristive_config.get("homeostasis_dropout", False)

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

        # Poole-Frenkel nonlinear forward map (manuscript Eq. forward-g).
        # When enabled, the per-synapse current is computed from the differentiable
        # PF model with (c, d_epsilon) sampled from the bivariate regression of
        # Eq. pf-regression, instead of the linear Ohmic product I = V * G.
        self.nonlinear = memristive_config.get("nonlinear", False)
        self.pf_slopes = memristive_config.get("pf_slopes")
        self.pf_intercepts = memristive_config.get("pf_intercepts")
        self.pf_covariance = memristive_config.get("pf_covariance")
        self.pf_stochastic = memristive_config.get("pf_stochastic", True)
        if self.nonlinear and (
            self.pf_slopes is None
            or self.pf_intercepts is None
            or self.pf_covariance is None
        ):
            raise ValueError(
                "nonlinear=True requires 'pf_slopes', 'pf_intercepts' and "
                "'pf_covariance' in memristive_config (see "
                "compute_PooleFrenkel_regression_parameters)."
            )
        # Precompute the Cholesky factor of the PF residual covariance once, so the
        # per-timestep PF sampling skips a redundant linalg.cholesky on a constant.
        self._pf_chol = (
            cholesky_factor_from_covariance(self.pf_covariance)
            if (self.nonlinear and self.pf_stochastic)
            else None
        )
        # When True, the PF device-to-device residual is sampled ONCE and frozen
        # for the run (a fixed per-synapse property), rather than re-rolled every
        # timestep. This is both faster (no per-step randn/matmul) and the more
        # physically faithful reading of device-to-device variability.
        self.pf_frozen_variability = memristive_config.get(
            "pf_frozen_variability", False
        )
        self._pf_residual = None

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

        # Disturb conductance based on the selected mode (stuck-at / device faults).
        # Homeostasis is applied at the spike level by the model's regulariser,
        # not here at the conductance level, so the two concerns stay separate.
        if self.disturb_conductance:
            if self.disturb_mode == "fixed":
                G, _mask = disturb_conductance_fixed(
                    G, self.G_on, true_probability=self.disturbance_probability
                )
            elif self.disturb_mode == "device":
                G = disturb_conductance_device(
                    G,
                    self.G_on,
                    self.G_off,
                    true_probability=self.disturbance_probability,
                )
            elif self.disturb_mode == "device_fixed":
                # Section 6.3 protocol: sample the stuck mask + D2D spread ONCE
                # (on first forward, when G's shape is known) and hold it fixed.
                if self._device_fault is None:
                    self._device_fault = sample_device_fault(
                        G.shape, self.G_on, self.G_off,
                        true_probability=self.disturbance_probability,
                        sigma_g=self.disturbance_sigma_g, device=G.device,
                        stuck_polarity=self.disturbance_stuck_polarity,
                    )
                G = apply_device_fault(G, self._device_fault)

        # Compute per-synapse current.
        # V: (batch, size_in+1); G: (size_in+1, 2*size_out) interleaved pos/neg.
        V_b = torch.unsqueeze(V, dim=-1)  # (batch, size_in+1, 1)
        G_b = torch.unsqueeze(G, dim=0)  # (1, size_in+1, 2*size_out)
        if self.nonlinear:
            # Poole-Frenkel nonlinear map (Eq. forward-g): sample (c, d_epsilon)
            # per synapse from its conductance, then I = PF(V, c, d_epsilon).
            if self.pf_frozen_variability and self.pf_stochastic:
                # Sample the device-to-device residual once (on first forward, when
                # G's broadcast shape is known) and reuse it thereafter.
                c, d_epsilon, res = sample_PooleFrenkel_parameters_torch(
                    G_b,
                    self.pf_slopes,
                    self.pf_intercepts,
                    self.pf_covariance,
                    stochastic=True,
                    cholesky_factor=self._pf_chol,
                    residual=self._pf_residual,
                    return_residual=True,
                )
                if self._pf_residual is None:
                    self._pf_residual = res.detach()
            else:
                c, d_epsilon = sample_PooleFrenkel_parameters_torch(
                    G_b,
                    self.pf_slopes,
                    self.pf_intercepts,
                    self.pf_covariance,
                    stochastic=self.pf_stochastic,
                    cholesky_factor=self._pf_chol,
                )
            I_ind = compute_PooleFrenkel_current_torch(V_b, c, d_epsilon)
        else:
            # Linear Ohmic map: I = V * G.
            I_ind = V_b * G_b
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
        self.disturbance_sigma_g = memristive_config.get("disturbance_sigma_g", 0.5)
        # Stuck-fault polarity for 'device_fixed' (see effects.sample_device_fault):
        # 'split' (symmetric), 'off' (dead synapses) or 'on' (saturated synapses).
        self.disturbance_stuck_polarity = memristive_config.get(
            "disturbance_stuck_polarity", "split"
        )
        self._device_fault = None
        self.homeostasis_dropout = memristive_config.get("homeostasis_dropout", False)

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

        # Poole-Frenkel nonlinear forward map for the convolution (beyond Joksas et al.,
        # who applied PF to dense layers only). A conv KERNEL weight is one physical
        # device reused across every spatial position, so the stuck-fault mask and the
        # PF device-to-device residual are sampled ONCE over the (K, 2*out_channels)
        # conductance grid and broadcast over batch and space.
        self.nonlinear = memristive_config.get("nonlinear", False)
        self.pf_slopes = memristive_config.get("pf_slopes")
        self.pf_intercepts = memristive_config.get("pf_intercepts")
        self.pf_covariance = memristive_config.get("pf_covariance")
        self.pf_stochastic = memristive_config.get("pf_stochastic", True)
        if self.nonlinear and (
            self.pf_slopes is None
            or self.pf_intercepts is None
            or self.pf_covariance is None
        ):
            raise ValueError(
                "nonlinear=True requires 'pf_slopes', 'pf_intercepts' and "
                "'pf_covariance' in memristive_config (see "
                "compute_PooleFrenkel_regression_parameters)."
            )
        self._pf_chol = (
            cholesky_factor_from_covariance(self.pf_covariance)
            if (self.nonlinear and self.pf_stochastic)
            else None
        )
        self.pf_frozen_variability = memristive_config.get(
            "pf_frozen_variability", False
        )
        self._pf_residual = None
        # Memory bounding for the (B, L, K, 2*out) PF current tensor (the slow im2col
        # path): chunk over spatial L; pf_checkpoint bounds the backward peak.
        self.pf_chunk_L = memristive_config.get("pf_chunk_L", None)
        self.pf_checkpoint = memristive_config.get("pf_checkpoint", False)

    def _to_tuple(self, value):
        """Ensures kernel size and padding are tuples."""
        return value if isinstance(value, tuple) else (value, value)

    def forward(self, x):
        """
        Forward pass for the memristive convolutional layer.

        Args:
        - x (Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
        - Tensor: Output tensor (batch_size, out_channels, new_height, new_width).
        """
        if self.ideal:
            return nn.functional.conv2d(
                x, self.weight, self.bias, self.stride, self.padding
            )
        # Both the linear-memristive and PF-nonlinear paths go through the same im2col
        # current-accumulation pipeline; only the per-element current model differs.
        return self._memristive_conv(x)

    def _conductance_grid(self, device):
        """Map the kernel to the interleaved pos/neg conductance grid and apply the
        sampled-once stuck-fault realisation. Returns ``(G, k_I)`` with ``G`` of shape
        ``(K, 2*out_channels)`` (K = in_channels*kh*kw), mirroring the dense layer.
        Gradient flows through ``self.weight``."""
        Cout = self.out_channels
        K = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        Wmat = self.weight.reshape(Cout, K).t()            # (K, Cout)
        # max over the KERNEL only (bias added additively in output units), so a conv
        # weight and a dense weight of equal value see identical device physics and the
        # 1x1-conv == dense identity is exact.
        max_w = torch.max(torch.abs(self.weight))
        k_G = (self.G_on - self.G_off) / max_w
        k_I = self.k_V * k_G
        G_eff = k_G * Wmat
        zero = torch.tensor(0.0, device=device)
        G_pos = self.G_off + torch.maximum(G_eff, zero)
        G_neg = self.G_off - torch.minimum(G_eff, zero)
        G = torch.cat((G_pos[:, :, None], G_neg[:, :, None]), dim=-1).reshape(
            K, 2 * Cout
        )

        if self.disturb_conductance:
            if self.disturb_mode == "fixed":
                G, _mask = disturb_conductance_fixed(
                    G, self.G_on, true_probability=self.disturbance_probability
                )
            elif self.disturb_mode == "device":
                G = disturb_conductance_device(
                    G, self.G_on, self.G_off,
                    true_probability=self.disturbance_probability,
                )
            elif self.disturb_mode == "device_fixed":
                if self._device_fault is None:
                    self._device_fault = sample_device_fault(
                        G.shape, self.G_on, self.G_off,
                        true_probability=self.disturbance_probability,
                        sigma_g=self.disturbance_sigma_g, device=G.device,
                        stuck_polarity=self.disturbance_stuck_polarity,
                    )
                G = apply_device_fault(G, self._device_fault)
        return G, k_I

    def _pf_params(self, G):
        """Per-device PF parameters ``(c, d_epsilon)`` sampled once on the static
        ``(1, 1, K, 2*Cout)`` grid; frozen residual cached so later forwards reproduce
        it (the mean still carries gradient through ``G``)."""
        G_b = G[None, None, :, :]
        if self.pf_frozen_variability and self.pf_stochastic:
            c, d_epsilon, res = sample_PooleFrenkel_parameters_torch(
                G_b, self.pf_slopes, self.pf_intercepts, self.pf_covariance,
                stochastic=True, cholesky_factor=self._pf_chol,
                residual=self._pf_residual, return_residual=True,
            )
            if self._pf_residual is None:
                self._pf_residual = res.detach()
        else:
            c, d_epsilon = sample_PooleFrenkel_parameters_torch(
                G_b, self.pf_slopes, self.pf_intercepts, self.pf_covariance,
                stochastic=self.pf_stochastic, cholesky_factor=self._pf_chol,
            )
        return c, d_epsilon

    def _current_tile(self, V_tile, G, c, d_epsilon, k_I):
        """Per-spatial-tile current over (B, t, K, 2*Cout), summed over K, pos-neg, /k_I."""
        if self.nonlinear:
            I_ind = compute_PooleFrenkel_current_torch(V_tile, c, d_epsilon)
        else:
            I_ind = V_tile * G[None, None, :, :]
        I = torch.sum(I_ind, dim=2)
        return (I[..., 0::2] - I[..., 1::2]) / k_I

    def _memristive_conv(self, x):
        B = x.shape[0]
        Cout = self.out_channels
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        oh = (x.shape[2] + 2 * ph - kh) // sh + 1
        ow = (x.shape[3] + 2 * pw - kw) // sw + 1

        G, k_I = self._conductance_grid(x.device)          # (K, 2*Cout)
        c = d_epsilon = None
        if self.nonlinear:
            c, d_epsilon = self._pf_params(G)

            # Binary-spike fast path. When the input is a spike train (every element 0
            # or 1 -- MCSNN conv2 fed by a LIF spike output), PF(k_V*0)=0 and
            # PF(k_V*1)=I_on, so the per-synapse current folds into a fixed effective
            # kernel and the whole conv becomes a single F.conv2d: the exp/sqrt runs ONCE
            # over (K,2*Cout) instead of over (B,L,K,2*Cout). Measured ~19x faster than
            # the im2col path at conv2 scale. Gradient flows through I_on->(c,d_eps)->G
            # ->weight. The runtime guard MUST fall through to full PF whenever the
            # homeostasis regulariser has gated the input to continuous (0,1] values --
            # those are exactly the timesteps the homeostasis study measures.
            if torch.all((x == 0) | (x == 1)):
                K = G.shape[0]
                k_V_full = torch.full((1, 1, K, 2 * Cout), self.k_V,
                                      device=x.device, dtype=x.dtype)
                I_on = compute_PooleFrenkel_current_torch(
                    k_V_full, c, d_epsilon).reshape(K, 2 * Cout)
                W_eff = (I_on[:, 0::2] - I_on[:, 1::2]) / k_I       # (K, Cout)
                W_eff_kernel = W_eff.t().reshape(Cout, self.in_channels, kh, kw)
                out = nn.functional.conv2d(
                    x, W_eff_kernel, None, self.stride, self.padding)
                return out + self.bias.view(1, Cout, 1, 1)

        # im2col full-PF path: input patches (B, K, L), L = oh*ow output positions.
        patches = nn.functional.unfold(
            x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )
        V = self.k_V * patches                             # (B, K, L)
        L = V.shape[-1]

        Lc = self.pf_chunk_L or L
        outs = []
        for s in range(0, L, Lc):
            V_tile = V[:, :, s:s + Lc].permute(0, 2, 1).unsqueeze(-1)  # (B, t, K, 1)
            if self.pf_checkpoint and self.training and self.nonlinear:
                tile = torch.utils.checkpoint.checkpoint(
                    self._current_tile, V_tile, G, c, d_epsilon, k_I,
                    use_reentrant=False,
                )
            else:
                tile = self._current_tile(V_tile, G, c, d_epsilon, k_I)
            outs.append(tile)
        y_cols = torch.cat(outs, dim=1)                    # (B, L, Cout)

        out = y_cols.permute(0, 2, 1).reshape(B, Cout, oh, ow)
        return out + self.bias.view(1, Cout, 1, 1)


class HomeostaticRegulariser(nn.Module):
    r"""Firing-rate-dependent homeostatic regulariser (manuscript Eq. homeo).

    Modulates the effective conductance of a presynaptic line as a smooth,
    differentiable sigmoidal function of its firing rate ``r_in`` measured over a
    sliding window:

    .. math::
        \frac{G_\mathrm{eff}}{G_\mathrm{max}} = \frac{1}{1 + (r_\mathrm{in}/r_\mathrm{th})^{n}}

    applied as a multiplicative gate on the presynaptic spike train. The gate is
    ~1 for ``r_in << r_th`` (normal regime untouched) and decays as
    ``r_in ** -n`` for ``r_in >> r_th`` (hyperactive lines suppressed). It is
    row-local (depends only on the presynaptic line, no postsynaptic feedback) and
    differentiable in ``r_in``, ``r_th`` and ``n``, so it composes with
    surrogate-gradient BPTT and admits gradient-based hyperparameter tuning.

    This replaces the binary, non-differentiable :class:`HomeostasisDropout`,
    which is retained only for backward compatibility.

    Args:
        r_th (float): Firing-rate threshold at which depression begins, expressed
            as a fraction of time steps spiked in the window (``[0, 1]``).
        n (float): Sharpness exponent of the sigmoidal transition. Defaults to the
            parameter-free derived value ``n=1`` (single screening state, the steady
            state of the Ch5 fill-emission trap balance, Eq. 5.15); ``n>1`` is a
            cascade-sharpened probe, not the grounded default.
    """

    def __init__(self, r_th=0.8, n=1.0):
        super().__init__()
        self.r_th = r_th
        self.n = n

    def gate(self, spk_window):
        """Per-neuron conductance gate in ``(0, 1]`` from a window of spikes.

        Args:
        - spk_window (Tensor): Recent presynaptic spikes, shape
          ``(window, batch, *features)``.

        Returns:
        - Tensor: Gate of shape ``(batch, *features)``.
        """
        r_in = spk_window.mean(dim=0)
        ratio = torch.clamp(r_in / self.r_th, min=0.0)
        return 1.0 / (1.0 + ratio.pow(self.n))

    def forward(self, spk_window):
        """Scale the current-step spikes by the windowed homeostatic gate.

        Args:
        - spk_window (Tensor): Recent presynaptic spikes, shape
          ``(window, batch, *features)``; the last entry is the current step.

        Returns:
        - Tensor: Gated current-step spikes, shape ``(batch, *features)``.
        """
        return spk_window[-1] * self.gate(spk_window)


class HomeostasisDropout(nn.Module):
    def __init__(self):
        """
        A custom layer that drops the spike output if it has been spiking continuously
        over all time steps (across the entire sequence).

        Legacy binary mechanism, retained for backward compatibility. The
        firing-rate-dependent :class:`HomeostaticRegulariser` implements the
        manuscript's Eq. homeo and is the default in the models.
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
