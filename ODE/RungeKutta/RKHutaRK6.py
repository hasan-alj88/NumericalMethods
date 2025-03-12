from ODE.RungeKutta.RungeKuttaBase import RungeKuttaBase
import sympy as sp
import numpy as np

class HutaRK6(RungeKuttaBase):
    """
    Huta's 6th-order Runge-Kutta method

    0      | 0       0       0       0       0       0       0       0
    1/9    | 1/9     0       0       0       0       0       0       0
    1/6    | 1/24    1/8     0       0       0       0       0       0
    1/3    | 1/72    0       3/8     0       0       0       0       0
    1/2    | 91/576  -1/12   -105/128 607/768 0      0       0       0
    2/3    | -383/9000 -41/100 -975/128 18153/5000 89/625  0       0       0
    1/6    | 3/50    0       0       243/1000 51/250  3/100   0       0
    1      | 3/25    0       0       24/125   4/25    3/100   3/25    0
    --------------------------------------------------------------
          | 3/40    0       9/40    9/40     9/40    9/40    3/40    0
    """

    @property
    def b_vector(self) -> np.ndarray:
        return np.array([
            sp.Rational(3, 40), 0, sp.Rational(9, 40), sp.Rational(9, 40),
            sp.Rational(9, 40), sp.Rational(9, 40), sp.Rational(3, 40), 0
        ])

    @property
    def c_vector(self) -> np.ndarray:
        return np.array([
            0, sp.Rational(1, 9), sp.Rational(1, 6),
            sp.Rational(1, 3), sp.Rational(1, 2), sp.Rational(2, 3),
            sp.Rational(1, 6), 1
        ])

    @property
    def rk_matrix(self) -> np.ndarray:
        return np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [sp.Rational(1, 9), 0, 0, 0, 0, 0, 0, 0],
            [sp.Rational(1, 24), sp.Rational(1, 8), 0, 0, 0, 0, 0, 0],
            [sp.Rational(1, 72), 0, sp.Rational(3, 8), 0, 0, 0, 0, 0],
            [sp.Rational(91, 576), sp.Rational(-1, 12), sp.Rational(-105, 128), sp.Rational(607, 768), 0, 0, 0, 0],
            [sp.Rational(-383, 9000), sp.Rational(-41, 100), sp.Rational(-975, 128), sp.Rational(18153, 5000),
             sp.Rational(89, 625), 0, 0, 0],
            [sp.Rational(3, 50), 0, 0, sp.Rational(243, 1000), sp.Rational(51, 250), sp.Rational(3, 100), 0, 0],
            [sp.Rational(3, 25), 0, 0, sp.Rational(24, 125), sp.Rational(4, 25), sp.Rational(3, 100),
             sp.Rational(3, 25), 0]
        ])