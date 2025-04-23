from dataclasses import dataclass
from decimal import Decimal, getcontext

import numpy as np

from Core import Numerical
from StopConditions.StopIfEqual import StopIfEqual
from utils.DecimalTools import to_decimal_array


@dataclass
class LinearJacobiMethod(Numerical):
    coefficients: np.ndarray = None
    lhs: np.ndarray = None
    initial_guess: np.ndarray = None

    def __post_init__(self):
        # Set decimal precision based on tolerance
        min_tolerance = min(self.absolute_tolerance or np.inf, self.relative_tolerance or np.inf)
        if min_tolerance > 0:
            # Calculate required precision based on the tolerance
            precision = max(28, int(-Decimal(str(min_tolerance)).log10() + 10))
            getcontext().prec = precision
        else:
            # Default high precision if tolerance is 0
            getcontext().prec = 50

        n, m = self.coefficients.shape
        if n != m:
            raise ValueError("Linear Jacobi method only works for square matrices")
        self.lhs = self.lhs.reshape(n, 1)
        bn = self.lhs.shape[0]
        if bn != n:
            raise ValueError("LHS dimensions do not match")

        # Convert numpy arrays to Decimal arrays
        self.decimal_coefficients = to_decimal_array(self.coefficients)
        self.decimal_lhs = to_decimal_array(self.lhs)
        self.decimal_initial_guess = to_decimal_array(self.initial_guess)

        self.add_stop_condition(
            StopIfEqual(
                tracking='residual',
                value=0,
                absolute_tolerance=self.absolute_tolerance,
                relative_tolerance=self.relative_tolerance
            )
        )

    def calculate_residual(self, solution: np.ndarray) -> float:
        """Calculate residual using high precision Decimal arithmetic"""
        # Convert solution to Decimal if it's not already
        if not isinstance(solution.flat[0], Decimal):
            solution = to_decimal_array(solution)

        # Calculate Ax
        Ax = np.zeros(self.decimal_lhs.shape, dtype=object)
        for i in range(self.decimal_coefficients.shape[0]):
            sum_val = Decimal('0')
            for j in range(self.decimal_coefficients.shape[1]):
                sum_val += self.decimal_coefficients[i, j] * solution[j]
            Ax[i] = sum_val

        # Calculate residual vector (b - Ax)
        diff = np.array([self.decimal_lhs[i] - Ax[i] for i in range(len(self.decimal_lhs))])

        # Calculate Euclidean norm using Decimal
        sum_squares = Decimal('0')
        for val in diff.flatten():
            sum_squares += val * val

        return float(sum_squares.sqrt())

    @property
    def initial_state(self) -> dict:
        """Return initial state with Decimal-calculated residual"""
        return {
            **{f'x{i}': float(x) for i, x in enumerate(self.decimal_initial_guess.flatten(), start=1)},
            'residual': self.calculate_residual(self.decimal_initial_guess)
        }

    def step(self) -> dict:
        """
        Jacobi method with Decimal precision: x^{k+1}_i = (b_i - \sum_{j \neq i} a_{ij}x^k_j) / a_{ii}
        Returns: Dictionary with new state values and residual as floats
        """
        # Get current solution vector and convert to Decimal
        xk_float = np.array(list(self.history.last_state.values())[:-1])
        xk = to_decimal_array(xk_float)

        # Initialize next solution vector
        xkp1 = np.zeros(xk.shape, dtype=object)
        for i in range(len(xkp1)):
            xkp1[i] = Decimal('0')

        # Get right-hand side vector
        b = self.decimal_lhs.flatten()

        # Iterate through each equation
        for i in range(xk.shape[0]):
            # Get diagonal element
            a_ii = self.decimal_coefficients[i, i]

            # Calculate sum of a_{ij} * x^k_j for j \neq i using Decimal
            sum_term = Decimal('0')
            for j in range(self.decimal_coefficients.shape[1]):
                if j != i:  # Skip the diagonal element
                    sum_term += self.decimal_coefficients[i, j] * xk[j]

            # Calculate new value for x_i using the Jacobi formula with Decimal
            xkp1[i] = (b[i] - sum_term) / a_ii

        # Calculate residual with high precision
        residual = self.calculate_residual(xkp1)

        # Convert back to float for result dictionary
        result = {f'x{k}': x for k, x in enumerate(xkp1, start=1)}
        result['residual'] = residual

        return result