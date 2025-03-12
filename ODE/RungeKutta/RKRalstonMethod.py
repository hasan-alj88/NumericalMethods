from ODE.RungeKutta.RungeKuttaBase import RungeKuttaBase
import sympy as sp
import numpy as np

class RKRalstonMethod(RungeKuttaBase):
    """
    0   | 0     0
    2/3 | 2/3   0
    --------------
        | 1/4   3/4
    """
    @property
    def b_vector(self) -> np.ndarray:
        return np.array([sp.Rational(1, 4), sp.Rational(3, 4)])

    @property
    def c_vector(self) -> np.ndarray:
        return np.array([0, sp.Rational(2, 3)])

    @property
    def rk_matrix(self) -> np.ndarray:
        return np.array([
            [0, 0],
            [sp.Rational(2, 3), 0]
        ])