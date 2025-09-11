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

    # Incorporate device-to-device variability using lognormal distribution
    log_var = torch.var(torch.log(G + 1e-8))  # Add epsilon to avoid log(0)
    log_std = torch.sqrt(log_var)
    log_mu = torch.log(G + 1e-8) - (0.5 * log_var)
    variability_samples = torch.exp(torch.normal(mean=log_mu, std=log_std))

    # Combine stuck and non-stuck conductance values
    G[random_stuck_mask] = stuck_values[random_stuck_mask]
    G[~random_stuck_mask] = variability_samples[~random_stuck_mask]

    return G


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
