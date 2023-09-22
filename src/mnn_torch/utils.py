import torch
import numpy as np
import numpy.typing as npt

from scipy.stats import linregress


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


def predict_multivariate_linear_regression(x, slopes, intercepts, covariance_matrix):
    linear_fit = torch.einsum("i...,j->i...j", x, slopes) + torch.einsum(
        "i...,j->i...j", torch.ones(x.shape).to(x.device), intercepts
    )

    linear_deviations = torch.distributions.MultivariateNormal(
        loc=torch.zeros(2).to(x.device),
        scale_tril=torch.linalg.cholesky(covariance_matrix),
    ).sample(sample_shape=x.shape)

    linear_deviated_fit = linear_fit + linear_deviations
    deviated_fit = torch.einsum("...i->i...", linear_deviated_fit)

    return deviated_fit
