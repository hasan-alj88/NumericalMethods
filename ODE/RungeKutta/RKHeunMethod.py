from typing import Dict
import sympy as sp
import numpy as np

from ODE.RungeKutta.RungeKuttaBase import RungeKuttaBase


class RKHeunMethod(RungeKuttaBase):
    """
    0   | 0   0
    1   | 1   0
    -------------
        |1/2  1/2
    """
    @property
    def b_vector(self) -> np.ndarray:
        return np.array([sp.Rational(1, 2), sp.Rational(1, 2)])

    @property
    def c_vector(self) -> np.ndarray:
        return np.array([0, 1])

    @property
    def rk_matrix(self) -> np.ndarray:
        return np.array([
            [0, 0],
            [1, 0]
        ])
