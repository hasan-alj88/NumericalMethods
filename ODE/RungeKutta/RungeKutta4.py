from ODE.RungeKutta.RungeKuttaBase import RungeKuttaBase
import sympy as sp
import numpy as np

class RungeKutta4(RungeKuttaBase):
    """
    0   | 0       0       0       0
    1/2 | 1/2     0       0       0
    1/2 | 0       1/2     0       0
    1   | 0       0       1       0
    ---------------------------------
        | 1/6     1/3     1/3     1/6
    """
    @property
    def b_vector(self) -> np.ndarray:
        return np.array([
            sp.Rational(1, 6), sp.Rational(1, 3),
            sp.Rational(1, 3), sp.Rational(1, 6)])

    @property
    def c_vector(self) -> np.ndarray:
        return np.array([0, sp.Rational(1, 2), sp.Rational(1, 2), 1])

    @property
    def rk_matrix(self) -> np.ndarray:
        return np.array([
            [0, 0, 0, 0],
            [sp.Rational(1, 2), 0, 0, 0],
            [0, sp.Rational(1, 2), 0, 0],
            [0, 0, 1, 0]
        ])