import numpy as np
import numpy.typing as npt

def sort_multiple_arrays(key_lst: npt.NDArray, *other_lsts: npt.NDArray):
    """Sorts multiple arrays based on the values of `key_lst`."""
    sorted_idx = np.argsort(key_lst)
    sorted_key_lst = key_lst[sorted_idx]
    sorted_other_lsts = [other_lst[sorted_idx] for other_lst in other_lsts]
    return sorted_key_lst, *sorted_other_lsts
