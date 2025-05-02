import numpy as np
import sympy as sp


def lagrange_interpolation_polynomial(x: np.ndarray, y: np.ndarray) -> sp.Expr:
    symx = sp.Symbol('x')
    n = len(x)
    result = 0

    for i in range(n):
        # Start with y[i]
        term = y[i]

        # Build the Lagrange basis polynomial
        for j in range(n):
            if j != i:
                # Multiply by (x - x[j])/(x[i] - x[j])
                term *= (symx - x[j]) / (x[i] - x[j])

        result += term

    return sp.simplify(result)