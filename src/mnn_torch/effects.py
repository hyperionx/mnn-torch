import numpy as np
from scipy.optimize import curve_fit
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
    """
    mask = torch.rand(*G.shape, device=G.device) < true_probability
    G = torch.where(mask, fixed_conductance, G)
    return G


def disturb_conductance_device(G, G_on, G_off, R_on_log_std, R_off_log_std):
    """
    Disturbs the conductance based on device parameters such as G_on, G_off, and log standard deviations.

    Args:
    - G: The original conductance array.
    - G_on: The conductance when the device is in the "on" state.
    - G_off: The conductance when the device is in the "off" state.
    - R_on_log_std: The log standard deviation of the resistance when the device is "on".
    - R_off_log_std: The log standard deviation of the resistance when the device is "off".

    Returns:
    - G: The disturbed conductance array.
    """
    R = 1 / G  # Convert conductance to resistance
    R_on = 1 / G_on
    R_off = 1 / G_off

    # Interpolate log standard deviation for each resistance value
    log_std = np.interp(R, [R_on, R_off], [R_on_log_std, R_off_log_std])
    R_squared = np.power(R, 2)
    log_var = np.power(log_std, 2)

    # Compute variance and mean for lognormal distribution
    R_var = R_squared * (np.exp(log_var) - 1)
    log_mu = np.log(R_squared / np.sqrt(R_squared + R_var))

    # Generate new resistance values from lognormal distribution
    R = np.random.lognormal(mean=log_mu, sigma=log_std)

    # Convert back to conductance
    G = 1 / R
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
    I = (
        V_sign
        * c
        * V_abs
        * np.exp(
            const.elementary_charge
            * np.sqrt(const.elementary_charge * V_abs / (const.pi * d_epsilon) + 1e-18)
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
    fit_data = predict_with_multivariate_linear_regression(ln_R, slopes, intercepts, covariance_matrix)
    
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
        r = v[ref_idx] / i[ref_idx]
        R[idx] = r

        # Fit the Poole-Frenkel current model
        popt, _ = curve_fit(compute_PooleFrenkel_current, v, i, p0=[1e-5, 1e-16])
        c[idx] = popt[0]
        d_epsilon[idx] = popt[1]

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
    slopes, intercepts, covariance_matrix = compute_multivariate_linear_regression(x, y_1, y_2)

    return slopes, intercepts, covariance_matrix
