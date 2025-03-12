from ODE.RungeKutta.RungeKuttaBase import RungeKuttaBase
import sympy as sp
import numpy as np


class RKButcher5thOrder(RungeKuttaBase):
    """
    Butcher's (1964) fifth-order Runge-Kutta method

    0    | 0       0       0       0       0       0
    1/4  | 1/4     0       0       0       0       0
    1/4  | 1/8     1/8     0       0       0       0
    1/2  | 0       0       1/2     0       0       0
    3/4  | 3/16    -3/8    3/8     9/16    0       0
    1    | -3/7    8/7     6/7     -12/7   8/7     0
    -----------------------------------------------
         | 7/90    0       32/90   12/90   32/90   7/90
    """

    @property
    def b_vector(self) -> np.ndarray:
        return np.array([
            sp.Rational(7, 90), 0, sp.Rational(32, 90),
            sp.Rational(12, 90), sp.Rational(32, 90), sp.Rational(7, 90)
        ])

    @property
    def c_vector(self) -> np.ndarray:
        return np.array([
            0, sp.Rational(1, 4), sp.Rational(1, 4),
            sp.Rational(1, 2), sp.Rational(3, 4), 1
        ])

    @property
    def rk_matrix(self) -> np.ndarray:
        return np.array([
            [0, 0, 0, 0, 0, 0],
            [sp.Rational(1, 4), 0, 0, 0, 0, 0],
            [sp.Rational(1, 8), sp.Rational(1, 8), 0, 0, 0, 0],
            [0, 0, sp.Rational(1, 2), 0, 0, 0],
            [sp.Rational(3, 16), sp.Rational(-3, 8), sp.Rational(3, 8), sp.Rational(9, 16), 0, 0],
            [sp.Rational(-3, 7), sp.Rational(8, 7), sp.Rational(6, 7), sp.Rational(-12, 7), sp.Rational(8, 7), 0]
        ])