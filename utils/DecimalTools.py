from decimal import Decimal

import numpy as np


def to_decimal_array(arr):
    """Convert numpy array to array of Decimal objects for precision"""
    if isinstance(arr, np.ndarray):
        shape = arr.shape
        flat_arr = arr.flatten()
        decimal_flat = [Decimal(str(x)) for x in flat_arr]
        return np.array(decimal_flat).reshape(shape)
    return arr

def to_float_array(arr):
    """Convert array of Decimal objects back to float array"""
    if isinstance(arr, np.ndarray):
        return np.array([float(x) for x in arr.flatten()]).reshape(arr.shape)
    return arr