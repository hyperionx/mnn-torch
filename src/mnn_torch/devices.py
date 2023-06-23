import torch
import numpy as np
import scipy.constants as const
from scipy.io import loadmat
from scipy.optimize import curve_fit
from src.mnn_torch.utils import (
    sort_multiple_arrays,
    compute_multivariate_linear_regression_parameters,
)


def load_SiOx_multistate(data_path) -> np.array:

    experimental_data = loadmat(data_path)["data"]
    experimental_data = np.flip(experimental_data, axis=2)
    experimental_data = np.transpose(experimental_data, (1, 2, 0))
    experimental_data = experimental_data[:2, :, :]

    return experimental_data


def load_SiOx_curves(
    experimental_data, max_voltage=0.5, voltage_step=0.005, clean_data=True
):
    num_points = int(max_voltage / voltage_step) + 1

    data = experimental_data[:, :, :num_points]
    if clean_data:
        data = clean_experimental_data(data)
    voltages = data[1, :, :]
    currents = data[0, :, :]

    return voltages, currents


def clean_experimental_data(experimental_data, threshold=0.1):
    accepted_curves = []
    for idx in range(experimental_data.shape[1]):
        curve = experimental_data[0, idx, :]
        d2y_dx2 = np.gradient(np.gradient(curve))
        ratio = d2y_dx2 / np.mean(curve)
        if ratio.max() < threshold:
            accepted_curves.append(idx)

    return experimental_data[:, accepted_curves, :]


def compute_PooleFrenkel_current(V, c, d_epsilon):
    V = torch.unsqueeze(torch.tensor(V), -1)
    V_abs = torch.abs(V)
    V_sign = torch.sign(V)
    I = torch.mul(
        V_sign,
        c,
        V_abs,
        torch.exp(
            torch.div(
                torch.mul(
                    const.elementary_charge,
                    torch.sqrt(
                        torch.add(
                            torch.div(
                                torch.mul(const.elementary_charge, V_abs),
                                torch.mul(const.pi, d_epsilon),
                            ),
                            1e-18,
                        )
                    ),
                )
            ),
            torch.mul(const.Boltzmann, torch.add(const.zero_Celsius + 20.0)),
        ),
    )

    return I


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
    experimental_data, high_resistance_state=True, ratio=5
):
    V, I = load_SiOx_curves(experimental_data)
    R, c, d_epsilon, _, _ = compute_PooleFrenkel_relationship(V, I)
    R = torch.tensor(R, dtype=torch.float32)
    c = torch.tensor(c, dtype=torch.float32)
    d_epsilon = torch.tensor(d_epsilon, dtype=torch.float32)

    sep_idx = np.searchsorted(
        R, const.physical_constants["inverse of conductance quantum"][0]
    )

    if high_resistance_state:
        G_off = 1 / R[-1]
        G_on = G_off * ratio
        x = torch.log(R)[sep_idx:]
        y_1 = torch.log(c)[sep_idx:]
        y_2 = torch.log(d_epsilon)[sep_idx:]
    else:
        G_on = 1 / R[0]
        G_off = G_on / ratio
        x = torch.log(R)[:sep_idx]
        y_1 = torch.log(c)[:sep_idx]
        y_2 = torch.log(d_epsilon)[:sep_idx]

    (
        slopes,
        intercepts,
        covariance_matrix,
    ) = compute_multivariate_linear_regression_parameters(x, y_1, y_2)

    return G_off, G_on, slopes, intercepts, covariance_matrix
