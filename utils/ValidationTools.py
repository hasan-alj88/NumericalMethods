import inspect

import numpy as np
import sympy

from utils.ExceptionTools import IgnoreException


def is_nan(x) -> bool:
    with IgnoreException(TypeError):
        # Handle numpy arrays and values
        if isinstance(x, np.ndarray):
            return np.any(np.isnan(x))
        if np.isnan(x):
            return True

        # Handle sympy objects
        if isinstance(x, sympy.Basic):
            x: np.ndarray = np.array(x)
            return np.any(np.isnan(x))
    return x != x

def function_arg_count(func: callable) -> int:
    if func is None:
        raise ValueError(f'function {func.__name__} must be provided')
    elif not callable(func):
        raise ValueError(f'function must be callable. Got: {type(func)}')

    derivative_function_signature = inspect.signature(func)
    non_default_args = [
        p for p in derivative_function_signature.parameters.values()
        if p.default is inspect.Parameter.empty and
           p.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    ]
    return len(non_default_args)