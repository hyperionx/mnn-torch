import torch
import numpy as np

from scipy.optimize import curve_fit
import scipy.constants as const

from mnn_torch.devices import load_SiOx_curves
from mnn_torch.utils import (
    sort_multiple_arrays,
    compute_multivariate_linear_regression_parameters,
    predict_multivariate_linear_regression,
)


def disturb_conductance(G, fixed_conductance, true_probability):
    mask = torch.rand(G.shape) < true_probability
    G = torch.where(mask, fixed_conductance, G)


def compute_PooleFrenkel_current(V, c, d_epsilon):
    # TODO: convert to Torch
    # if torch.is_tensor(V):
    #     V = V.cpu().detach().numpy()
    #     V = np.expand_dims(V, axis=-1)
    #     c = c.cpu().detach().numpy()
    #     d_epsilon = d_epsilon.cpu().detach().numpy()

    V_abs = np.absolute(V)
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
    R = 1 / G
    ln_R = torch.log(R)

    fit_data = predict_multivariate_linear_regression(
        ln_R, slopes, intercepts, covariance_matrix
    )
    c = torch.exp(fit_data[0])
    d_epsilon = torch.exp(fit_data[1])
    I_ind = compute_PooleFrenkel_current(V, c, d_epsilon)

    # Add currents along bit lines
    I = np.sum(I_ind, axis=1)

    return I, I_ind


def compute_PooleFrenkel_relationship(V, I, voltage_step=0.005, ref_voltage=0.1):
    num_curves = V.shape[0]
    R = np.zeros(num_curves)
    c = np.zeros(num_curves)
    d_epsilon = np.zeros(num_curves)
    ref_idx = int(ref_voltage / voltage_step)

    for idx in range(num_curves):
        v = V[idx, :]
        i = I[idx, :]

        r = v[ref_idx] / i[ref_idx]
        R[idx] = r
        popt, pcov = curve_fit(compute_PooleFrenkel_current, v, i, p0=[1e-5, 1e-16])
        c[idx] = popt[0]
        d_epsilon[idx] = popt[1]

    R, c, d_epsilon, V, I = sort_multiple_arrays(R, c, d_epsilon, V, I)
    return R, c, d_epsilon, V, I


def compute_PooleFrenkel_parameters(
    experimental_data, high_resistance_state=False, ratio=5
):
    V, I = load_SiOx_curves(experimental_data)
    R, c, d_epsilon, _, _ = compute_PooleFrenkel_relationship(V, I)
    R = torch.tensor(R, dtype=torch.float32)
    c = torch.tensor(c, dtype=torch.float32)
    d_epsilon = torch.tensor(d_epsilon, dtype=torch.float32)


    if high_resistance_state:
        G_off = 1 / R[-1]
        G_on = G_off * ratio
    else:
        G_on = 1 / R[0]
        G_off = G_on / ratio

    return G_off, G_on, R, c, d_epsilon


def compute_PooleFrenkel_regression_parameters(R, c, d_epsilon, high_resistance_state=False):
    sep_idx = np.searchsorted(
        R, const.physical_constants["inverse of conductance quantum"][0]
    )

    if high_resistance_state:
        x = torch.log(R)[sep_idx:]
        y_1 = torch.log(c)[sep_idx:]
        y_2 = torch.log(d_epsilon)[sep_idx:]
    else:
        x = torch.log(R)[:sep_idx]
        y_1 = torch.log(c)[:sep_idx]
        y_2 = torch.log(d_epsilon)[:sep_idx]

    (
        slopes,
        intercepts,
        covariance_matrix,
    ) = compute_multivariate_linear_regression_parameters(x, y_1, y_2)

    return slopes, intercepts, covariance_matrix
