from ODE.RungeKutta.RungeKuttaBase import RungeKuttaBase
import sympy as sp
import numpy as np

class RKBogackiShampine(RungeKuttaBase):
    """
    Bogacki-Shampine method - 3rd order method with embedded 2nd order solution
    Used in MATLAB's ode23

    0   | 0       0       0       0
    1/2 | 1/2     0       0       0
    3/4 | 0       3/4     0       0
    1   | 2/9     1/3     4/9     0
    -------------------------
    3rd | 2/9     1/3     4/9     0
    2nd | 7/24    1/4     1/3     1/8
    """
    @property
    def b_vector(self) -> np.ndarray:
        # 3rd-order coefficients
        return np.array([
            sp.Rational(2, 9), sp.Rational(1, 3),
            sp.Rational(4, 9), 0
        ])

    @property
    def c_vector(self) -> np.ndarray:
        return np.array([
            0, sp.Rational(1, 2), sp.Rational(3, 4), 1
        ])

    @property
    def rk_matrix(self) -> np.ndarray:
        return np.array([
            [0, 0, 0, 0],
            [sp.Rational(1, 2), 0, 0, 0],
            [0, sp.Rational(3, 4), 0, 0],
            [sp.Rational(2, 9), sp.Rational(1, 3), sp.Rational(4, 9), 0]
        ])