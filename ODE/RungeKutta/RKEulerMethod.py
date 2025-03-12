import numpy as np

from ODE.RungeKutta.RungeKuttaBase import RungeKuttaBase


class RKEulerMethod(RungeKuttaBase):
    """
    0   |   0
    ---------
        |   1
    """
    @property
    def b_vector(self) -> np.ndarray:
        return np.array([1])

    @property
    def c_vector(self) -> np.ndarray:
        return np.array([0])

    @property
    def rk_matrix(self) -> np.ndarray:
        return np.array([[1]])