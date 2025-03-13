from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from ODE.RungeKutta.RungeKuttaBase import RungeKuttaBase


@dataclass
class CustomRungeKutta(RungeKuttaBase):
    """
    Custom Runge-Kutta method that accepts user-provided Butcher tableau components.

    This class allows you to create any explicit Runge-Kutta method by providing:
    - rk_matrix: The A matrix in the Butcher tableau
    - b_vector: The weights for the stages (must sum to 1)
    - c_vector: The node points

    Example usage:
    ```python
    # Define RK4 method manually
    a_matrix = np.array([
        [0, 0, 0, 0],
        [1/2, 0, 0, 0],
        [0, 1/2, 0, 0],
        [0, 0, 1, 0]
    ])
    b = np.array([1/6, 1/3, 1/3, 1/6])
    c = np.array([0, 1/2, 1/2, 1])

    rk4 = CustomRungeKutta(
        derivative_function=my_ode_function,
        y0=1.0,
        t0=0.0,
        t_final=10.0,
        h=0.1,
        rk_matrix=a_matrix,
        b_vector=b,
        c_vector=c,
        method_name="RK4"
    )
    ```
    """
    # These parameters come from RungeKuttaBase
    # derivative_function: Callable[[float, float], float] = field(default=None)
    # y0: float = 0.0
    # t0: float = 0.0
    # t_final: float = field(default=None)
    # h: float = 0.01

    # Custom parameters
    _rk_matrix: np.ndarray = field(default=None)
    _b_vector: np.ndarray = field(default=None)
    _c_vector: np.ndarray = field(default=None)
    method_name: str = field(default="Custom RK Method")

    def __post_init__(self):
        # Convert arrays to sympy Rational type if they contain floats
        if self._rk_matrix is not None and np.issubdtype(self._rk_matrix.dtype, np.floating):
            self._rk_matrix = self._float_array_to_rational(self._rk_matrix)

        if self._b_vector is not None and np.issubdtype(self._b_vector.dtype, np.floating):
            self._b_vector = self._float_array_to_rational(self._b_vector)

        if self._c_vector is not None and np.issubdtype(self._c_vector.dtype, np.floating):
            self._c_vector = self._float_array_to_rational(self._c_vector)

        # Now call the parent's post_init
        super().__post_init__()

    @staticmethod
    def _float_array_to_rational(arr):
        """Convert a numpy array of floats to sympy Rational objects with reasonable precision"""
        result = np.empty(arr.shape, dtype=object)
        for idx in np.ndindex(arr.shape):
            val = arr[idx]
            # Convert float to a rational approximation
            rational = sp.Rational(str(val))
            result[idx] = rational
        return result

    @property
    def rk_matrix(self) -> np.ndarray:
        return self._rk_matrix

    @property
    def b_vector(self) -> np.ndarray:
        return self._b_vector

    @property
    def c_vector(self) -> np.ndarray:
        return self._c_vector

    def __str__(self):
        return f"{self.method_name} (Custom Runge-Kutta)"