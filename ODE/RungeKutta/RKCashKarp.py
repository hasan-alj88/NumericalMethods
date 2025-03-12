from ODE.RungeKutta.RungeKuttaBase import RungeKuttaBase
import sympy as sp
import numpy as np


class RKCashKarp(RungeKuttaBase):
    """
    Cash-Karp method - 5th order method with embedded 4th order solution

    0      | 0       0       0       0       0       0
    1/5    | 1/5     0       0       0       0       0
    3/10   | 3/40    9/40    0       0       0       0
    3/5    | 3/10    -9/10   6/5     0       0       0
    1      | -11/54  5/2     -70/27  35/27   0       0
    7/8    | 1631/55296 175/512 575/13824 44275/110592 253/4096 0
    -------------------------------------------------------------
    5th    | 37/378  0       250/621 125/594 0       512/1771
    4th    | 2825/27648 0     18575/48384 13525/55296 277/14336 1/4
    """

    @property
    def b_vector(self) -> np.ndarray:
        # Using 5th-order coefficients
        return np.array([
            sp.Rational(37, 378), 0, sp.Rational(250, 621),
            sp.Rational(125, 594), 0, sp.Rational(512, 1771)
        ])

    @property
    def c_vector(self) -> np.ndarray:
        return np.array([
            0, sp.Rational(1, 5), sp.Rational(3, 10),
            sp.Rational(3, 5), 1, sp.Rational(7, 8)
        ])

    @property
    def rk_matrix(self) -> np.ndarray:
        return np.array([
            [0, 0, 0, 0, 0, 0],
            [sp.Rational(1, 5), 0, 0, 0, 0, 0],
            [sp.Rational(3, 40), sp.Rational(9, 40), 0, 0, 0, 0],
            [sp.Rational(3, 10), sp.Rational(-9, 10), sp.Rational(6, 5), 0, 0, 0],
            [sp.Rational(-11, 54), sp.Rational(5, 2), sp.Rational(-70, 27), sp.Rational(35, 27), 0, 0],
            [sp.Rational(1631, 55296), sp.Rational(175, 512), sp.Rational(575, 13824),
             sp.Rational(44275, 110592), sp.Rational(253, 4096), 0]
        ])

