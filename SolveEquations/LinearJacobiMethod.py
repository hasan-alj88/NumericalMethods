from dataclasses import dataclass
from typing import Dict, Any
import sympy as sp
from Core import Numerical
from StopConditions.StopIfEqual import StopIfEqual


@dataclass
class LinearJacobiMethod(Numerical):
    coefficients: sp.Matrix = None
    lhs: sp.Matrix = None
    initial_guess: sp.Matrix = None

    def __post_init__(self):
        n,m = self.coefficients.shape
        if n!=m:
            raise ValueError("Linear Jacobi method only works for square matrices")
        self.lhs = self.lhs.reshape(n,1)
        bn = self.coefficients.shape[0]
        if bn != n:
            raise ValueError("LHS dimensions do not match")

        self.add_stop_condition(
            StopIfEqual(
                tracking='residual',value=0, absolute_tolerance=self.absolute_tolerance, relative_tolerance=self.relative_tolerance)
        )

    def calculate_residual(self, solution: sp.Matrix) -> float:
        diff = self.lhs - self.coefficients @ solution
        return diff.norm().evalf()



    @property
    def initial_state(self) -> dict:
        return {
            **{f'x{i}': x for i, x in enumerate(self.initial_guess.flat(), start=1)},
            'residual': self.calculate_residual(self.initial_guess)
        }

    def step(self) -> Dict[str, Any]:
        """
        x^{k+1}_i = \frac{1}{a_{ii}} (b_i - \sum_{j \neq i} a_{ij}x^k_j)
        :return: Dictionary with new state values and residual
        """
        # Get current solution vector
        xk = sp.Matrix(list(self.history.last_state.values())[:-1])

        # Initialize next solution vector
        xkp1 = sp.zeros(xk.rows, 1)

        # Get right-hand side vector
        b = self.lhs

        # Iterate through each equation
        for i in range(xk.rows):
            # Get diagonal element
            a_ii = self.coefficients[i, i]

            # Calculate sum of a_ij * x_j for j ≠ i
            sum_term = 0
            for j in range(self.coefficients.cols):
                if j != i:
                    sum_term += self.coefficients[i, j] * xk[j]

            # Calculate new value for x_i
            xkp1[i] = (b[i] - sum_term) / a_ii

        # Create result dictionary
        result = {f'x{k}': x for k, x in enumerate(xkp1, start=1)}
        result['residual'] = self.calculate_residual(xkp1)

        return result