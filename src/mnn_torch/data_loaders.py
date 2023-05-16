import numpy as np
from scipy.io import loadmat

def SiOx_multistate_loader(data_path) -> np.array:

    experimental_data = loadmat(data_path)["data"]
    experimental_data = np.flip(experimental_data, axis=2)
    experimental_data = np.transpose(experimental_data, (1, 2, 0))
    experimental_data = experimental_data[:2, :, :]

    return experimental_data
