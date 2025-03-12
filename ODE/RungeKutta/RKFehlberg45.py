from ODE.RungeKutta.RungeKuttaBase import RungeKuttaBase
import sympy as sp
import numpy as np


class RKFehlberg45(RungeKuttaBase):
    """
    Fehlberg 4(5) method - 4th order method with embedded 5th order solution

    0      | 0       0       0       0       0       0
    1/4    | 1/4     0       0       0       0       0
    3/8    | 3/32    9/32    0       0       0       0
    12/13  | 1932/2197 -7200/2197 7296/2197 0       0       0
    1      | 439/216 -8      3680/513 -845/4104 0       0
    1/2    | -8/27   2       -3544/2565 1859/4104 -11/40  0
    --------------------------------------------------------------
    4th    | 25/216  0       1408/2565 2197/4104 -1/5    0
    5th    | 16/135  0       6656/12825 28561/56430 -9/50 2/55
    """
    @property
    def b_vector(self) -> np.ndarray:
        # Using 4th-order coefficients
        return np.array([
            sp.Rational(25, 216), 0, sp.Rational(1408, 2565),
            sp.Rational(2197, 4104), sp.Rational(-1, 5), 0
        ])

    @property
    def c_vector(self) -> np.ndarray:
        return np.array([
            0, sp.Rational(1, 4), sp.Rational(3, 8),
            sp.Rational(12, 13), 1, sp.Rational(1, 2)
        ])

    @property
    def rk_matrix(self) -> np.ndarray:
        return np.array([
            [0, 0, 0, 0, 0, 0],
            [sp.Rational(1, 4), 0, 0, 0, 0, 0],
            [sp.Rational(3, 32), sp.Rational(9, 32), 0, 0, 0, 0],
            [sp.Rational(1932, 2197), sp.Rational(-7200, 2197), sp.Rational(7296, 2197), 0, 0, 0],
            [sp.Rational(439, 216), -8, sp.Rational(3680, 513), sp.Rational(-845, 4104), 0, 0],
            [sp.Rational(-8, 27), 2, sp.Rational(-3544, 2565), sp.Rational(1859, 4104), sp.Rational(-11, 40), 0]
        ])
