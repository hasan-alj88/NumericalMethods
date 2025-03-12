from ODE.RungeKutta.RungeKuttaBase import RungeKuttaBase
import sympy as sp
import numpy as np

class Verner6thOrder(RungeKuttaBase):
    """
    Verner's efficient 6th-order Runge-Kutta method

    0      | 0       0       0       0       0       0       0       0
    1/6    | 1/6     0       0       0       0       0       0       0
    4/15   | 4/75    16/75   0       0       0       0       0       0
    2/5    | 2/75    0       8/75    0       0       0       0       0
    1      | 0       0       0       5/6     0       0       0       0
    2/3    | 23/616  0       0       177/1408 -9/32  0       0       0
    1/3    | 4/39    0       0       0       3/13    12/39   0       0
    1      | 7/90    0       0       16/45   2/15    16/45   7/90    0
    --------------------------------------------------------------
          | 7/90    0       0       16/45   2/15    16/45   7/90    0
    """

    @property
    def b_vector(self) -> np.ndarray:
        return np.array([
            sp.Rational(7, 90), 0, 0, sp.Rational(16, 45),
            sp.Rational(2, 15), sp.Rational(16, 45),
            sp.Rational(7, 90), 0
        ])

    @property
    def c_vector(self) -> np.ndarray:
        return np.array([
            0, sp.Rational(1, 6), sp.Rational(4, 15),
            sp.Rational(2, 5), 1, sp.Rational(2, 3),
            sp.Rational(1, 3), 1
        ])

    @property
    def rk_matrix(self) -> np.ndarray:
        return np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [sp.Rational(1, 6), 0, 0, 0, 0, 0, 0, 0],
            [sp.Rational(4, 75), sp.Rational(16, 75), 0, 0, 0, 0, 0, 0],
            [sp.Rational(2, 75), 0, sp.Rational(8, 75), 0, 0, 0, 0, 0],
            [0, 0, 0, sp.Rational(5, 6), 0, 0, 0, 0],
            [sp.Rational(23, 616), 0, 0, sp.Rational(177, 1408), sp.Rational(-9, 32), 0, 0, 0],
            [sp.Rational(4, 39), 0, 0, 0, sp.Rational(3, 13), sp.Rational(12, 39), 0, 0],
            [sp.Rational(7, 90), 0, 0, sp.Rational(16, 45), sp.Rational(2, 15), sp.Rational(16, 45), sp.Rational(7, 90),
             0]
        ])

