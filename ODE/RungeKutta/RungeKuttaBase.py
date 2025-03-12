from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import sympy

from Core import Numerical
from StopConditions.StopIfGreaterThan import StopIfGreaterThan
from utils.ValidationTools import function_arg_count, raise_value_error_if_none


@dataclass
class RungeKuttaBase(Numerical, ABC):
    """
    Runge-Kutta methods for solving first-order ODEs:

    $$\frac{dy}{dt} = f(y, t)$$

    The general form of a Runge-Kutta method:

    $$y_{n+1} = y_n + h \sum_{i=1}^s b_i k_i$$

    where each stage $k_i$ is calculated as:

    $$k_i = f\left(t_n + c_i h, y_n + h \sum_{j=1}^{i-1} a_{ij} k_j\right)$$

    The Butcher tableau represents the method:
    The Runge-Kutta coefficient matrix (A in the Butcher tableau).
    Must be a square matrix of shape (stage_order, stage_order).

    $$
    \begin{array}{c|cccc}
    c_1 & a_{11} & a_{12} & \cdots & a_{1s} \\
    c_2 & a_{21} & a_{22} & \cdots & a_{2s} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    c_s & a_{s1} & a_{s2} & \cdots & a_{ss} \\
    \hline
    & b_1 & b_2 & \cdots & b_s
    \end{array}
    $$
    """
    derivative_function: Callable[[float, float], float] = field(default=None)
    y0: float = 0.0
    t0: float = 0.0
    t_final: float = field(default=None)
    h: float = 0.01

    def __post_init__(self):
        if function_arg_count(self.derivative_function) != 2:
            raise ValueError(f'Derivative function must take 2 arguments f(y,t), '
                             f'not {function_arg_count(self.derivative_function)}')

        raise_value_error_if_none(
            dict(t0=self.t0, t_final=self.t_final, h=self.h, y0=self.y0)
        )

        if self.h <= 0:
            raise ValueError(f'h({self.h}) must be greater than 0')
        if self.t_final <= self.t0:
            raise ValueError(f't_final({self.t_final}) must be greater than t0 ({self.t0})')
        if not np.isclose(self.b_vector.sum(), 1.0):
            raise ValueError(f"b_vector must sum to 1. Got {self.rk_matrix.sum()}")

        if self.b_vector.shape != (self.stage_order,):
            raise ValueError(f"b_vector must be a {self.stage_order}x1 vector")
        if self.c_vector.shape != (self.stage_order,):
            raise ValueError(f"c_vector must be a {self.stage_order}x1 vector")
        self.validate_butcher_tableau()

        self.add_stop_condition(StopIfGreaterThan(tracking='t', threshold=self.t_final-2*self.h))


    @property
    def butcher_tableau(self) -> str:
        """
        Construct the Butcher tableau in LaTeX format with a vertical line after the first column 
        and a horizontal line drawn before the bottom row.
        """
        tableau = np.hstack((self.c_vector.reshape(-1, 1), self.rk_matrix))

        rows = []
        for row in tableau:
            rows.append(" & ".join(map(sympy.latex, row)))
    
        butcher_tableau_latex = "\\begin{array}{c|c" + "c" * (self.stage_order - 1) + "}\n"
        butcher_tableau_latex += " \\\\\n".join(rows)
        butcher_tableau_latex += " \\\\\n\\hline\n"
        butcher_tableau_latex += "0 & " + " & ".join(map(str, self.b_vector)) + " \\\\\n"
        butcher_tableau_latex += "\\end{array}"
        return butcher_tableau_latex

    def validate_butcher_tableau(self):
        """
        Validate dimensions of rk_matrix, b_vector, and c_vector for compatibility.
        """
        if self.rk_matrix.shape[0] != self.c_vector.size:
            raise ValueError(
                f"rk_matrix row count ({self.rk_matrix.shape[0]}) must match c_vector length ({self.c_vector.size})"
            )

        if self.rk_matrix.shape[1] != self.b_vector.size:
            raise ValueError(
                f"rk_matrix column count ({self.rk_matrix.shape[1]}) must match b_vector length ({self.b_vector.size})"
            )
    @property
    def stage_order(self) -> int:
        return len(self.b_vector)

    @abstractmethod
    @property
    def rk_matrix(self) -> np.ndarray:
        pass

    @abstractmethod
    @property
    def b_vector(self) -> np.ndarray:
        pass

    @abstractmethod
    @property
    def c_vector(self) -> np.ndarray:
        pass

    def compute_k_values(self, ti, yi):
        """
        Compute all k values for the current step
        """
        k_values = np.zeros(self.stage_order)

        for s in range(self.stage_order):
            t_s = ti + self.c_vector[s] * self.h
            y_s = yi

            # Sum up the contributions from previous stages
            for j in range(s):
                y_s += self.h * self.rk_matrix[s, j] * k_values[j]

            k_values[s] = self.derivative_function(y_s, t_s)

        return k_values

    @property
    def initial_state(self) -> dict:
        return dict(
            y=self.y0,
            t=self.t0,
            dy_dt=self.derivative_function(self.y0, self.t0)
        )

    def step(self) -> dict:
        yi = self.history['y']
        ti = self.history['t']

        k_values = self.compute_k_values(ti, yi)
        yi1 = yi + self.h * np.sum(self.b_vector * k_values)
        ti1 = ti + self.h

        # Convert symbolic expressions to floats if needed
        yi1 = float(yi1.evalf()) if isinstance(yi1, sympy.Expr) else yi1
        ti1 = float(ti1.evalf()) if isinstance(ti1, sympy.Expr) else ti1

        # Calculate derivative only once after converting to floats
        dy_dt1 = self.derivative_function(yi1, ti1)
        # Convert derivative to float if it's a symbolic expression
        dy_dt1 = float(dy_dt1.evalf()) if isinstance(dy_dt1, sympy.Expr) else dy_dt1

        return dict(
            y=yi1,
            t=ti1,
            dy_dt=dy_dt1
        )