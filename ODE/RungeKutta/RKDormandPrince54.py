from ODE.RungeKutta.RungeKuttaBase import RungeKuttaBase
import sympy as sp
import numpy as np

class RKDormandPrince54(RungeKuttaBase):
    """
    Dormand-Prince 5(4) method - 5th order method with embedded 4th order solution
    Often used in adaptive step size implementations (like MATLAB's ode45)

    0      | 0          0           0            0           0           0          0
    1/5    | 1/5        0           0            0           0           0          0
    3/10   | 3/40       9/40        0            0           0           0          0
    4/5    | 44/45      -56/15      32/9         0           0           0          0
    8/9    | 19372/6561 -25360/2187 64448/6561   -212/729    0           0          0
    1      | 9017/3168  -355/33     46732/5247   49/176      -5103/18656 0          0
    1      | 35/384     0           500/1113     125/192     -2187/6784  11/84      0
    ------------------------------------------------------------------------------
    5th    | 35/384     0           500/1113     125/192     -2187/6784  11/84      0
    4th    | 5179/57600 0           7571/16695   393/640     -92097/339200 187/2100 1/40
    """

    @property
    def b_vector(self) -> np.ndarray:
        # We'll use the 5th-order coefficients for the main solution
        return np.array([
            sp.Rational(35, 384), 0, sp.Rational(500, 1113),
            sp.Rational(125, 192), sp.Rational(-2187, 6784),
            sp.Rational(11, 84), 0
        ])

    @property
    def c_vector(self) -> np.ndarray:
        return np.array([
            0, sp.Rational(1, 5), sp.Rational(3, 10),
            sp.Rational(4, 5), sp.Rational(8, 9), 1, 1
        ])

    @property
    def rk_matrix(self) -> np.ndarray:
        return np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [sp.Rational(1, 5), 0, 0, 0, 0, 0, 0],
            [sp.Rational(3, 40), sp.Rational(9, 40), 0, 0, 0, 0, 0],
            [sp.Rational(44, 45), sp.Rational(-56, 15), sp.Rational(32, 9), 0, 0, 0, 0],
            [sp.Rational(19372, 6561), sp.Rational(-25360, 2187), sp.Rational(64448, 6561), sp.Rational(-212, 729), 0,
             0, 0],
            [sp.Rational(9017, 3168), sp.Rational(-355, 33), sp.Rational(46732, 5247), sp.Rational(49, 176),
             sp.Rational(-5103, 18656), 0, 0],
            [sp.Rational(35, 384), 0, sp.Rational(500, 1113), sp.Rational(125, 192), sp.Rational(-2187, 6784),
             sp.Rational(11, 84), 0]
        ])


