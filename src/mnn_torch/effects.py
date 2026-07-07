import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde, truncnorm
import scipy.constants as const
import torch

from mnn_torch.devices import load_SiOx_curves
from mnn_torch.utils import (
    sort_multiple_arrays,
    compute_multivariate_linear_regression,
    predict_with_multivariate_linear_regression,
)


def disturb_conductance_fixed(G, fixed_conductance, true_probability=0.5):
    """
    Disturbs the conductance by randomly replacing values with a fixed conductance.

    Args:
    - G: The original conductance array (tensor).
    - fixed_conductance: The fixed conductance value to replace randomly selected entries.
    - true_probability: The probability of replacing an entry with the fixed conductance.

    Returns:
    - G: The disturbed conductance array (tensor).
    - mask: The boolean mask indicating which values were replaced.
    """
    mask = torch.rand(*G.shape, device=G.device) < true_probability
    G = torch.where(mask, fixed_conductance, G)
    return G, mask


def disturb_conductance_device(G, G_on, G_off, true_probability=0.1):
    """
    Models stuck devices by probabilistically fixing their conductance to G_off or G_on.

    Args:
    - G (torch.Tensor): Original conductance tensor.
    - G_on (float or torch.Tensor): Maximum conductance value.
    - G_off (float or torch.Tensor): Minimum conductance value.
    - true_probability (float): Proportion of devices to be stuck (0 to 1).

    Returns:
    - torch.Tensor: Updated conductance tensor with probabilistic stuck behavior.
    """
    device = G.device
    G_on = torch.tensor(G_on, device=device, dtype=torch.float32)
    G_off = torch.tensor(G_off, device=device, dtype=torch.float32)

    # Calculate conductance range and median range
    conductance_range = G_on - G_off
    median_range = torch.median(conductance_range)

    # Identify potentially stuck devices based on range criteria
    stuck_mask = (G - G_off).abs() < (0.5 * median_range)

    # Generate a PDF for stuck values using KDE
    G_flat = G[~stuck_mask].detach().cpu().numpy()  # Non-stuck conductance values
    kde = gaussian_kde(G_flat)
    kde_samples = kde.resample(len(G_flat)).flatten()

    # Apply truncated normal distribution to mitigate bias near zero
    a, b = (0 - kde_samples.mean()) / kde_samples.std(), float("inf")
    truncated_samples = truncnorm(
        a, b, loc=kde_samples.mean(), scale=kde_samples.std()
    ).rvs(len(G_flat))
    truncated_samples = torch.tensor(
        truncated_samples, device=device, dtype=torch.float32
    )

    # Map stuck devices probabilistically to G_off or G_on
    stuck_values = torch.where(torch.rand_like(G, device=device) < 0.5, G_off, G_on)

    # Generate a random mask to select which devices are stuck
    random_stuck_mask = torch.rand_like(G, device=device) < true_probability

    # Incorporate device-to-device variability using a lognormal distribution
    # (Eq. lognormal). The lognormal model is defined on positive conductance, so
    # operate on |G| to stay valid when G carries a sign (e.g. conv kernels where
    # G = G_pos - G_neg); a positive floor avoids log(0)/NaN that would make the
    # sampling std non-finite. The original sign is restored after sampling.
    G_sign = torch.sign(G)
    G_mag = torch.clamp(torch.abs(G), min=1e-8)
    log_var = torch.clamp(torch.var(torch.log(G_mag)), min=0.0)
    log_std = torch.sqrt(log_var)
    log_mu = torch.log(G_mag) - (0.5 * log_var)
    # Sample per element: draw standard normals and scale by the (scalar) std so
    # the call is shape-safe across backends (MPS rejects a 0-dim std against a
    # full-shaped mean), then exponentiate. Equivalent to N(log_mu, log_std**2).
    noise = torch.randn_like(log_mu) * log_std
    variability_samples = G_sign * torch.exp(log_mu + noise)

    # Combine stuck and non-stuck conductance values out-of-place: stuck devices
    # take the fixed stuck value, the rest take the lognormal-perturbed value.
    # An out-of-place select (rather than in-place index assignment) is required
    # because G may be part of an autograd graph when the conductance feeds the
    # differentiable Poole-Frenkel forward map.
    G = torch.where(random_stuck_mask, stuck_values, variability_samples)

    return G


def sample_device_fault(shape, G_on, G_off, true_probability=0.1, sigma_g=0.5,
                        device="cpu", generator=None, stuck_polarity="split"):
    """Draw a *fixed* device-fault realisation, once, for a conductance array.

    This implements the Chapter-6 / Section-6.3 fault protocol, in which the
    stuck-device pattern and the device-to-device (D2D) programming spread are
    sampled ONCE per training run and then held fixed, exactly as a physically
    programmed crossbar would behave: a device that is stuck, or programmed off
    its target, stays that way for the rest of training. This is distinct from
    :func:`disturb_conductance_device`, which re-rolls a fresh random fault on
    every forward pass and therefore injects a new, destabilising perturbation at
    every time step rather than a fixed hardware fault.

    The realisation is independent of the live weight values: it records *which*
    devices are stuck (``stuck_mask``), the conductance they are stuck at
    (``stuck_values``), and a per-device multiplicative lognormal D2D factor
    (``d2d_factor``, Eq. lognormal). It is applied later by
    :func:`apply_device_fault` to whatever conductance the current weights map to.

    The ``stuck_polarity`` argument follows the Joksas et al. nonideality
    taxonomy (``StuckAtGOff`` / ``StuckAtGOn``), which separates the two failure
    modes that the symmetric default blends together:

    * ``"off"`` -- stuck-at-G_off: dead / open synapses (cannot conduct). The
      presynaptic line goes silent; a firing-rate homeostat has nothing to act
      on, so this is the natural *negative control* for the homeostasis claim.
    * ``"on"``  -- stuck-at-G_on: shorted / saturated synapses that drive their
      postsynaptic line hyperactive. This is exactly the regime the sigmoidal
      firing-rate regulariser (Eq. homeo) is designed to suppress, so it is
      where homeostatic recovery should be largest.
    * ``"split"`` (default) -- each stuck device is G_off or G_on with equal
      probability, the original symmetric behaviour.

    Args:
    - shape (tuple): Shape of the conductance tensor the fault applies to.
    - G_on, G_off (float): Conductance bounds; stuck devices take one of these.
    - true_probability (float): Fraction of devices that are stuck.
    - sigma_g (float): Lognormal width of the D2D programming spread (Eq. lognormal).
    - device: Torch device for the realisation tensors.
    - generator (torch.Generator, optional): For reproducible realisations.
    - stuck_polarity (str): ``"split"`` | ``"off"`` | ``"on"`` (see above).

    Returns:
    - dict: {"stuck_mask", "stuck_values", "d2d_factor"} torch tensors of `shape`,
      all detached (a fixed hardware realisation, not a function of the weights).
    """
    G_on_t = torch.as_tensor(G_on, device=device, dtype=torch.float32)
    G_off_t = torch.as_tensor(G_off, device=device, dtype=torch.float32)

    def _rand(s):
        return torch.rand(s, device=device, generator=generator)

    def _randn(s):
        return torch.randn(s, device=device, generator=generator)

    stuck_mask = _rand(shape) < true_probability
    if stuck_polarity == "off":
        stuck_values = G_off_t.expand(shape).clone()
    elif stuck_polarity == "on":
        stuck_values = G_on_t.expand(shape).clone()
    elif stuck_polarity == "split":
        stuck_values = torch.where(_rand(shape) < 0.5, G_off_t, G_on_t)
    else:
        raise ValueError(
            f"stuck_polarity must be 'split', 'off' or 'on'; got {stuck_polarity!r}"
        )
    # Multiplicative lognormal D2D spread, mean-preserving (median at the target):
    # G_observed = G_target * exp(N(0, sigma_g^2)).
    d2d_factor = torch.exp(sigma_g * _randn(shape))
    return {
        "stuck_mask": stuck_mask.detach(),
        "stuck_values": stuck_values.detach(),
        "d2d_factor": d2d_factor.detach(),
    }


def apply_device_fault(G, fault):
    """Apply a *fixed* device-fault realisation (from :func:`sample_device_fault`).

    Stuck devices take their fixed stuck conductance; all other devices keep their
    programmed conductance scaled by the fixed per-device lognormal D2D factor. The
    operation is out-of-place so it is safe inside an autograd graph (the surviving
    devices' gradient still flows through ``G``), while the stuck devices and the
    D2D factors are constants that do not move during training.

    Args:
    - G (torch.Tensor): Live conductance tensor from the current weights.
    - fault (dict): Realisation from :func:`sample_device_fault`.

    Returns:
    - torch.Tensor: Conductance with the fixed fault realisation applied.
    """
    sign = torch.sign(G)
    perturbed = sign * torch.abs(G) * fault["d2d_factor"]
    return torch.where(fault["stuck_mask"], fault["stuck_values"], perturbed)


def compute_PooleFrenkel_current_torch(V, c, d_epsilon):
    r"""Differentiable Poole-Frenkel current (Torch), the forward map of the manuscript.

    Implements Eq. ``forward-g`` / Eq. ``pf``:

    .. math::
        I = c\, V \exp\!\left( \frac{2e}{k_B T} \sqrt{\frac{e |V|}{4\pi d\varepsilon}} \right)

    at ``T = 20 degC``, with the sign of ``V`` preserved so the map is valid for
    both bias polarities. All operations are Torch ops, so gradients flow through
    ``V``, ``c`` and ``d_epsilon``; this is the differentiable replacement for the
    NumPy :func:`compute_PooleFrenkel_current` used in offline fitting.

    Args:
        V (Tensor): Voltage.
        c (Tensor): Conductance-like prefactor (broadcastable to ``V``).
        d_epsilon (Tensor): Effective oxide-thickness--permittivity product
            (broadcastable to ``V``).

    Returns:
        Tensor: Poole-Frenkel current, same shape as the broadcast of the inputs.
    """
    e = const.elementary_charge
    kT = const.Boltzmann * (const.zero_Celsius + 20.0)
    V_abs = torch.abs(V)
    V_sign = torch.sign(V)
    # Manuscript Eq. pf: (2e/kT) * sqrt(e|V| / (4 pi d_epsilon)). This is
    # algebraically identical to the NumPy reference's (e/kT)*sqrt(e|V|/(pi d_eps))
    # since 2*sqrt(1/4) = 1. Clamp the radicand for numerical stability.
    term = torch.clamp(e * V_abs / (4.0 * np.pi * d_epsilon), min=1e-18)
    exponent = (2.0 * e / kT) * torch.sqrt(term)
    # Clamp the exponent: when a synapse's sampled conductance falls outside the
    # fitted resistance range, the regression can predict an unphysically small
    # d_epsilon and the raw exp() overflows to inf. Bounding the exponent keeps
    # the forward map finite and differentiable without distorting the in-range
    # response (exp(50) already exceeds any realistic current scale here).
    exponent = torch.clamp(exponent, max=50.0)
    return V_sign * c * V_abs * torch.exp(exponent)


def cholesky_factor_from_covariance(covariance_matrix, device="cpu", dtype=torch.float32):
    """Cholesky factor ``L`` of the PF residual covariance, with PD jitter.

    Precomputing this once (e.g. at layer construction) avoids recomputing a
    ``torch.linalg.cholesky`` on the constant covariance at every forward pass,
    which is the dominant redundant cost in the per-timestep PF sampling.
    """
    Sigma = torch.as_tensor(covariance_matrix, device=device, dtype=dtype)
    jitter = 1e-12 * torch.eye(2, device=device, dtype=dtype)
    return torch.linalg.cholesky(Sigma + jitter)


def sample_PooleFrenkel_parameters_torch(
    G, slopes, intercepts, covariance_matrix, stochastic=True, generator=None,
    cholesky_factor=None, residual=None, return_residual=False,
):
    r"""Sample per-synapse Poole-Frenkel parameters ``(c, d_epsilon)`` from conductance.

    Implements the bivariate regression of the manuscript (Eq. ``pf-regression``):

    .. math::
        [\ln c, \ln(d\varepsilon)]^\top = -\mathbf{m}\,\ln R + \mathbf{b} + \mathbf{E},
        \quad \mathbf{E} \sim \mathcal{N}_2(0, \Sigma)

    with ``R = 1/G``. The regression mean depends on the (trainable) conductance
    ``G`` and so carries gradient; the Gaussian residual ``E`` is treated as
    injected noise and is detached from the graph (stop-gradient), matching the
    manuscript's "Gradient flow" description. Returns ``c`` and ``d_epsilon`` as
    Torch tensors on ``G``'s device.

    Args:
        G (Tensor): Per-synapse conductance, any shape.
        slopes (array-like): Regression slopes ``m`` for ``[ln c, ln d_epsilon]``.
        intercepts (array-like): Regression intercepts ``b``.
        covariance_matrix (array-like): Residual covariance ``Sigma`` (2x2).
        stochastic (bool): If True, add the ``N(0, Sigma)`` residual; if False,
            return the regression mean only (deterministic).
        generator (torch.Generator, optional): RNG for reproducible sampling.
        cholesky_factor (Tensor, optional): Precomputed Cholesky factor of the
            residual covariance (see :func:`cholesky_factor_from_covariance`). When
            given, the per-call ``cholesky`` is skipped; ``covariance_matrix`` is
            then unused. Pass this from a layer that caches it once at construction.
        residual (Tensor, optional): A frozen ``(*G.shape, 2)`` device-to-device
            residual to reuse instead of drawing a fresh one. Models the variability
            as a fixed per-synapse property (sampled once per run), which also avoids
            the per-timestep ``randn``/matmul. ``cholesky_factor`` is then unused.
        return_residual (bool): If True, also return the residual that was used, so
            a caller can cache it for subsequent forward passes.

    Returns:
        tuple[Tensor, Tensor]: ``(c, d_epsilon)`` with the same shape as ``G``; or
        ``(c, d_epsilon, residual)`` when ``return_residual`` is True.
    """
    device = G.device
    dtype = G.dtype
    m = torch.as_tensor(slopes, device=device, dtype=dtype)
    b = torch.as_tensor(intercepts, device=device, dtype=dtype)

    ln_R = torch.log(1.0 / torch.clamp(G, min=1e-12))
    # mean of [ln c, ln d_epsilon]: shape (*G.shape, 2).
    # NOTE: slopes/intercepts come from scipy.linregress (compute_multivariate_
    # linear_regression), whose convention is y = slope*x + intercept, so the
    # prediction uses +slopes here. This is the signed form of the manuscript's
    # Eq. pf-regression (-m ln R + b), where the code's slope already carries the
    # sign; the +convention yields physical d_epsilon ~1e-18 matching the fit.
    mean = ln_R.unsqueeze(-1) * m + b

    drawn_residual = None
    if stochastic:
        if residual is not None:
            # Frozen device-to-device residual (sampled once per run): the
            # variability is a fixed property of each fabricated synapse, so it is
            # not re-rolled every timestep. The weight-dependent mean still updates,
            # so gradients and training-time conductance changes are preserved.
            drawn_residual = residual.to(device=device, dtype=dtype)
        else:
            # Use a precomputed Cholesky factor when supplied (constant across
            # forward passes); otherwise compute it from the covariance. Computing
            # it once at layer construction removes a linalg op from the timestep loop.
            if cholesky_factor is not None:
                L = cholesky_factor.to(device=device, dtype=dtype)
            else:
                L = cholesky_factor_from_covariance(covariance_matrix, device, dtype)
            z = torch.randn(*G.shape, 2, device=device, dtype=dtype, generator=generator)
            drawn_residual = z @ L.T
        ln_params = mean + drawn_residual.detach()  # stop-gradient on the noise
    else:
        ln_params = mean

    c = torch.exp(ln_params[..., 0])
    d_epsilon = torch.exp(ln_params[..., 1])
    if return_residual:
        return c, d_epsilon, drawn_residual
    return c, d_epsilon


def compute_PooleFrenkel_current(V, c, d_epsilon):
    """
    Computes the Poole-Frenkel current for a given voltage.

    Args:
    - V: The voltage array.
    - c: A coefficient related to the current.
    - d_epsilon: A parameter related to the material's dielectric properties.

    Returns:
    - I: The calculated current array.
    """
    V_abs = np.abs(V)
    V_sign = np.sign(V)
    term = np.maximum(const.elementary_charge * V_abs / (const.pi * d_epsilon), 1e-18)
    I = (
        V_sign
        * c
        * V_abs
        * np.exp(
            const.elementary_charge
            * np.sqrt(term)
            / (const.Boltzmann * (const.zero_Celsius + 20.0))
        )
    )
    return I


def compute_PooleFrenkel_total_current(V, G, slopes, intercepts, covariance_matrix):
    """
    Computes the total Poole-Frenkel current, including deviations based on the regression parameters.

    Args:
    - V: The voltage array.
    - G: The conductance array.
    - slopes: The slopes from the regression model.
    - intercepts: The intercepts from the regression model.
    - covariance_matrix: The covariance matrix for the regression.

    Returns:
    - I: The total current.
    - I_ind: The individual current components.
    """
    R = 1 / G
    ln_R = np.log(R)

    # Predict regression results for ln(R)
    fit_data = predict_with_multivariate_linear_regression(
        ln_R, slopes, intercepts, covariance_matrix
    )

    # Extract coefficients for the Poole-Frenkel equation
    c = np.exp(fit_data[0])
    d_epsilon = np.exp(fit_data[1])

    # Compute individual currents
    I_ind = compute_PooleFrenkel_current(V, c, d_epsilon)

    # Sum currents along the bit lines
    I = np.sum(I_ind, axis=1)
    return I, I_ind


def compute_PooleFrenkel_relationship(V, I, voltage_step=0.005, ref_voltage=0.1):
    """
    Computes the relationship between voltage and current for the Poole-Frenkel model.

    Args:
    - V: Voltage array.
    - I: Current array.
    - voltage_step: Step size for voltage.
    - ref_voltage: The reference voltage to compute resistance.

    Returns:
    - R, c, d_epsilon: The computed resistance, c, and d_epsilon values.
    - V, I: The sorted voltage and current arrays.
    """
    num_curves = V.shape[0]
    R = np.zeros(num_curves)
    c = np.zeros(num_curves)
    d_epsilon = np.zeros(num_curves)

    ref_idx = int(ref_voltage / voltage_step)

    for idx in range(num_curves):
        v = V[idx, :]
        i = I[idx, :]

        # Compute resistance at the reference voltage
        if i[ref_idx] == 0:  # Avoid division by zero
            R[idx] = np.inf
        else:
            R[idx] = v[ref_idx] / i[ref_idx]

        # Fit the Poole-Frenkel current model
        try:
            popt, _ = curve_fit(
                compute_PooleFrenkel_current,
                v,
                i,
                p0=[1e-5, 1e-16],
                bounds=([1e-8, 1e-18], [1e-3, 1e-14]),
                maxfev=10000,  # Increase iterations if needed
            )
            c[idx] = popt[0]
            d_epsilon[idx] = popt[1]
        except RuntimeError as e:
            print(f"Curve fit failed for curve {idx}: {e}")
            c[idx], d_epsilon[idx] = np.nan, np.nan

    # Sort the results
    R, c, d_epsilon, V, I = sort_multiple_arrays(R, c, d_epsilon, V, I)
    return R, c, d_epsilon, V, I


def compute_PooleFrenkel_parameters(
    experimental_data, high_resistance_state=False, ratio=5
):
    """
    Computes the Poole-Frenkel model parameters based on experimental data.

    Args:
    - experimental_data: The data used for the calculations.
    - high_resistance_state: Boolean indicating whether to use a high resistance state.
    - ratio: The ratio of the conductance in "on" and "off" states.

    Returns:
    - G_off, G_on: The conductance in the "off" and "on" states.
    - R, c, d_epsilon: The computed resistance, c, and d_epsilon values.
    """
    V, I = load_SiOx_curves(experimental_data)
    R, c, d_epsilon, _, _ = compute_PooleFrenkel_relationship(V, I)

    # Convert results to float32 for consistency
    R = R.astype(np.float32)
    c = c.astype(np.float32)
    d_epsilon = d_epsilon.astype(np.float32)

    # Compute G_on and G_off based on the resistance state
    if high_resistance_state:
        G_off = 1 / R[-1]
        G_on = G_off * ratio
    else:
        G_on = 1 / R[0]
        G_off = G_on / ratio

    return G_off, G_on, R, c, d_epsilon


def compute_PooleFrenkel_regression_parameters(
    R, c, d_epsilon, high_resistance_state=False
):
    """
    Computes the regression parameters for the Poole-Frenkel model.

    Args:
    - R: The resistance array.
    - c: The coefficient array.
    - d_epsilon: The dielectric constant array.
    - high_resistance_state: Boolean indicating the state.

    Returns:
    - slopes, intercepts, covariance_matrix: The regression parameters.
    """
    sep_idx = np.searchsorted(
        R, const.physical_constants["inverse of conductance quantum"][0]
    )

    if high_resistance_state:
        x = np.log(R[sep_idx:])
        y_1 = np.log(c[sep_idx:])
        y_2 = np.log(d_epsilon[sep_idx:])
    else:
        x = np.log(R[:sep_idx])
        y_1 = np.log(c[:sep_idx])
        y_2 = np.log(d_epsilon[:sep_idx])

    # Compute regression parameters
    slopes, intercepts, covariance_matrix = compute_multivariate_linear_regression(
        x, y_1, y_2
    )

    return slopes, intercepts, covariance_matrix
