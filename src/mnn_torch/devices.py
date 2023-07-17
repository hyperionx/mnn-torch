import numpy as np
from scipy.io import loadmat


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
