import numpy as np
from scipy.stats import linregress


def sort_multiple_arrays(key_array: np.ndarray, *arrays_to_sort: np.ndarray):
    """
    Sorts multiple arrays based on the values in `key_array`.

    Args:
    - key_array: The array used as a sorting key.
    - arrays_to_sort: The arrays that will be sorted in the same way as `key_array`.

    Returns:
    - The sorted key_array, followed by all the other sorted arrays.
    """
    # Get the indices that would sort the key array
    sorted_indices = np.argsort(key_array)

    # Sort all arrays using the sorted indices
    sorted_key_array = key_array[sorted_indices]
    sorted_arrays = [array[sorted_indices] for array in arrays_to_sort]

    return sorted_key_array, *sorted_arrays


def compute_multivariate_linear_regression(x: np.ndarray, *y_arrays: np.ndarray):
    """
    Computes the multivariate linear regression parameters (slopes, intercepts, residuals).

    Args:
    - x: The independent variable.
    - y_arrays: Multiple dependent variables.

    Returns:
    - slopes: The slope of the regression for each dependent variable.
    - intercepts: The intercept of the regression for each dependent variable.
    - covariance_matrix: The covariance matrix of the residuals across all dependent variables.
    """
    slopes = []
    intercepts = []
    residuals_list = []

    for y in y_arrays:
        # Perform linear regression on x and y
        regression_result = linregress(x, y)

        # Store regression parameters
        slopes.append(regression_result.slope)
        intercepts.append(regression_result.intercept)

        # Compute residuals (actual - predicted values)
        predicted_y = regression_result.slope * x + regression_result.intercept
        residuals = y - predicted_y
        residuals_list.append(residuals)

    # Convert lists to numpy arrays
    slopes_array = np.array(slopes, dtype=np.float32)
    intercepts_array = np.array(intercepts, dtype=np.float32)
    residuals_array = np.array(residuals_list, dtype=np.float32)

    # Compute the covariance matrix of the residuals
    covariance_matrix = np.cov(residuals_array)

    return slopes_array, intercepts_array, covariance_matrix


def predict_with_multivariate_linear_regression(
    x: np.ndarray,
    slopes: np.ndarray,
    intercepts: np.ndarray,
    covariance_matrix: np.ndarray,
):
    """
    Predicts values using a multivariate linear regression model, including deviations based on residuals.

    Args:
    - x: The input values to predict for.
    - slopes: The slopes for the linear regression model.
    - intercepts: The intercepts for the linear regression model.
    - covariance_matrix: The covariance matrix of the residuals.

    Returns:
    - deviated_fit: The predicted values with added deviations from the covariance matrix.
    """
    # Calculate the linear fit (dot product between x and slopes, plus intercepts)
    linear_fit = np.einsum("i...,j->i...j", x, slopes) + np.einsum(
        "i...,j->i...j", np.ones(x.shape), intercepts
    )

    # Generate deviations based on the covariance matrix
    linear_deviations = np.random.multivariate_normal(
        mean=np.zeros(2), cov=covariance_matrix, size=x.shape[0]
    )

    # Add deviations to the linear fit to get the final predicted values
    deviated_fit = linear_fit + linear_deviations

    # Summarize and return the result
    return np.sum(deviated_fit, axis=-1)
