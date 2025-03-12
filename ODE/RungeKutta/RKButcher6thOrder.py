from ODE.RungeKutta.RungeKuttaBase import RungeKuttaBase
import sympy as sp
import numpy as np


class RKButcher6thOrder(RungeKuttaBase):
    """
    Butcher's 6th-order Runge-Kutta method

    0      | 0       0       0       0       0       0       0
    1/3    | 1/3     0       0       0       0       0       0
    2/3    | 0       2/3     0       0       0       0       0
    1/3    | 1/12    1/3     -1/12   0       0       0       0
    5/6    | 25/48   -55/24  35/48   15/8    0       0       0
    1/6    | 3/20    -11/24  -1/8    1/2     1/10    0       0
    1      | -261/260 33/13  43/156  -118/39 32/195  80/39   0
    --------------------------------------------------------------
          | 13/200  0       11/40   11/40   4/25    4/25    13/200
    """

    @property
    def b_vector(self) -> np.ndarray:
        return np.array([
            sp.Rational(13, 200), 0, sp.Rational(11, 40),
            sp.Rational(11, 40), sp.Rational(4, 25),
            sp.Rational(4, 25), sp.Rational(13, 200)
        ])

    @property
    def c_vector(self) -> np.ndarray:
        return np.array([
            0, sp.Rational(1, 3), sp.Rational(2, 3),
            sp.Rational(1, 3), sp.Rational(5, 6),
            sp.Rational(1, 6), 1
        ])

    @property
    def rk_matrix(self) -> np.ndarray:
        return np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [sp.Rational(1, 3), 0, 0, 0, 0, 0, 0],
            [0, sp.Rational(2, 3), 0, 0, 0, 0, 0],
            [sp.Rational(1, 12), sp.Rational(1, 3), sp.Rational(-1, 12), 0, 0, 0, 0],
            [sp.Rational(25, 48), sp.Rational(-55, 24), sp.Rational(35, 48), sp.Rational(15, 8), 0, 0, 0],
            [sp.Rational(3, 20), sp.Rational(-11, 24), sp.Rational(-1, 8), sp.Rational(1, 2), sp.Rational(1, 10), 0, 0],
            [sp.Rational(-261, 260), sp.Rational(33, 13), sp.Rational(43, 156), sp.Rational(-118, 39),
             sp.Rational(32, 195), sp.Rational(80, 39), 0]
        ])



