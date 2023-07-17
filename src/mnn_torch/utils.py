import torch
import numpy as np
import numpy.typing as npt

from scipy.stats import linregress
from scipy.optimize import curve_fit
import scipy.constants as const

from mnn_torch.devices import load_SiOx_curves


def sort_multiple_arrays(key_lst: npt.NDArray, *other_lsts: npt.NDArray):
    """Sorts multiple arrays based on the values of `key_lst`."""
    sorted_idx = np.argsort(key_lst)
    sorted_key_lst = key_lst[sorted_idx]
    sorted_other_lsts = [other_lst[sorted_idx] for other_lst in other_lsts]
    return sorted_key_lst, *sorted_other_lsts


def compute_multivariate_linear_regression_parameters(x, *y):
    slopes: list[float] = []
    intercepts: list[float] = []
    residuals_list: list[float] = []

    for y_i in y:
        result = linregress(x, y_i)
        slopes.append(result.slope)
        intercepts.append(result.intercept)

        y_pred = result.slope * x + result.intercept
        residuals = y_i - y_pred
        residuals_list.append(residuals.tolist())

    slopes = torch.tensor(slopes, dtype=torch.float32)
    intercepts = torch.tensor(intercepts, dtype=torch.float32)
    residuals_list = torch.tensor(residuals_list, dtype=torch.float32)
    covariance_matrix = torch.cov(residuals_list)

    return slopes, intercepts, covariance_matrix


def compute_PooleFrenkel_current(V, c, d_epsilon):
    V = torch.unsqueeze(torch.tensor(V), -1)
    V_abs = torch.abs(V)
    V_sign = torch.sign(V)
    (I,) = (
        V_sign
        * c
        * V_abs
        * torch.exp(
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
                ),
                torch.mul(const.Boltzmann, torch.add(const.zero_Celsius, 20.0)),
            ),
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
        selected_v = V[idx, :]
        selected_i = I[idx, :]

        selected_r = selected_v[ref_idx] / selected_i[ref_idx]
        R[idx] = selected_r
        popt, pcov = curve_fit(
            fit_model_parameters, selected_v, selected_i, p0=[1e-5, 1e-16]
        )
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


def fit_model_parameters(
    V: npt.NDArray[np.float64], c: float, d_times_perm: float
) -> npt.NDArray[np.float64]:
    V = torch.tensor(V)
    I = compute_PooleFrenkel_current(V, c, d_times_perm)
    I = I.numpy()[:, 0]

    return I
