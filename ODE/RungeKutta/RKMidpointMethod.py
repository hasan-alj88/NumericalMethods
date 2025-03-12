import numpy as np

from ODE.RungeKutta.RungeKuttaBase import RungeKuttaBase
import sympy as sp

class RKMidpointMethod(RungeKuttaBase):
    """
    0   | 0     0
    1/2 | 1/2   0
    --------------
        | 0     1
    """
    @property
    def b_vector(self) -> np.array:
        return np.array([0, 1])

    @property
    def c_vector(self) -> np.array:
        return np.array([0, sp.Rational(1, 2)])

    @property
    def rk_matrix(self) -> np.ndarray:
        return np.array([
            [0, 0],
            [sp.Rational(1, 2), 0]
        ])