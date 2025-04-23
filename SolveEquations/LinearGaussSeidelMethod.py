from decimal import Decimal
from typing import Dict

import numpy as np

from SolveEquations.LinearJacobiMethod import LinearJacobiMethod
from utils.DecimalTools import to_decimal_array


class LinearGaussSeidelMethod(LinearJacobiMethod):
    def step(self) -> Dict[str, float]:
        """
        Gauss-Seidel method with Decimal precision:
        x^{k+1}_i = (b_i - \sum_{j<i} a_{ij}x^{k+1}_j - \sum_{j>i} a_{ij}x^k_j) / a_{ii}

        Returns: Dictionary with new state values and residual as floats
        """
        # Get current solution vector and convert to Decimal
        xk_float = np.array(list(self.history.last_state.values())[:-1])
        xk = to_decimal_array(xk_float)

        # Initialize next solution vector with existing values
        xkp1 = xk.copy()

        # Get right-hand side vector
        b = self.decimal_lhs.flatten()

        # Iterate through each equation
        for i in range(xk.shape[0]):
            # Get diagonal element
            a_ii = self.decimal_coefficients[i, i]

            # Calculate two sums using Decimal:
            # 1. Sum for j < i using updated values (x^{k+1})
            # 2. Sum for j > i using previous values (x^k)
            sum_term = Decimal('0')

            # Use already updated values for j < i
            for j in range(i):
                sum_term += self.decimal_coefficients[i, j] * xkp1[j]

            # Use previous iteration values for j > i
            for j in range(i + 1, self.decimal_coefficients.shape[1]):
                sum_term += self.decimal_coefficients[i, j] * xk[j]

            # Calculate new value for x_i using the Gauss-Seidel formula with Decimal
            xkp1[i] = (b[i] - sum_term) / a_ii

        # Calculate residual with high precision
        residual = self.calculate_residual(xkp1)

        # Convert back to float for result dictionary
        result = {f'x{k}': float(x) for k, x in enumerate(xkp1, start=1)}
        result['residual'] = residual

        return result