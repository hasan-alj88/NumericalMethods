from itertools import product

import numpy as np
import sympy as sp


def newton_coefficients(x:np.ndarray, y: np.ndarray, order:int)->np.ndarray:
    y = y[:order+1]
    x = x[:order]
    if order < 1:
        raise ValueError("Order must be greater than 0")
    # divide difference table
    dd = np.zeros((order+1,order+1))
    dd[:,0] = y
    for j in range(1, order+1):
        for i in range(order-j):
            dd[i, j] = (dd[i + 1, j - 1] - dd[i, j - 1]) / (x[i + j] - x[i])
    return dd[0,:order+1]


def newton_interpolation_polynomial(x: np.ndarray, y: np.ndarray, order:int) -> sp.Expr:
    symx = sp.Symbol('x')
    coefs = newton_coefficients(x, y, order)

    # Start with the first coefficient
    polynomial = coefs[0]

    # Build the polynomial term by term
    product_term = 1
    for xi, b in zip(x, coefs[1:]):
        product_term *= (symx - xi)
        polynomial += b * product_term

    # Simplify the expression
    polynomial = sp.simplify(polynomial)

    return polynomial


def lagrange_interpolation_polynomial(x: np.ndarray, y: np.ndarray) -> sp.Expr:
    symx = sp.Symbol('x')
    n = len(x)
    s = 0
    for i in range(n):
        product_term = y[i]
        for j in range(n):
            if j != i:
                product_term *= (symx - x[j])
        s += product_term
    return sp.simplify(s)
