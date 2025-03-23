from typing import Tuple

import numpy as np
import sympy
from IPython.core.display import Markdown, Math
from IPython.display import display

from utils.LaTeXTools import numpy_to_latex_gauss, vector2latex


def forward_elimination(aug: sympy.Matrix, verbose:bool=False) -> sympy.Matrix:
    n = aug.rows
    aug = aug.copy()

    # Pivot row
    for k in range(n - 1):
        # Rows below pivot
        for i in range(k + 1, n):
            if aug[i, k] == 0:
                # Only process non-zero elements
                continue
            factor = aug[i, k] / aug[k, k]
            # Row operation Ri -> Ri - alpha Rk
            aug[i, k:] = aug[i, k:] - factor * aug[k, k:]
        if verbose: display(Math(numpy_to_latex_gauss(aug=aug)))
    return aug


def back_substitution(aug: sympy.Matrix) -> sympy.Matrix:
    n = aug.rows
    x = sympy.zeros(n, 1)  # Create a column vector

    for i in range(n - 1, -1, -1):  # Start from bottom row
        # Calculate the sum of known terms
        sum_known = sum([aug[i, j] * x[j, 0]for j in range(i + 1, n)])

        # Solve for the current variable
        x[i, 0] = (aug[i, n] - sum_known) / aug[i, i]

    return x
def forward_substitution(aug: sympy.Matrix) -> sympy.Matrix:
    n = aug.rows
    x = sympy.zeros(n, 1)

    for i in range(n):
        known_sum = sum([aug[i,j]*x[j] for j in range(i + 1)])
        x[i] = (aug[i, n] - known_sum) / aug[i, i]

    return x


def validate_ax_b_shapes(a: sympy.Matrix, b: sympy.Matrix) -> Tuple[sympy.Matrix, sympy.Matrix]:
    # Validate Coefficient matrix shape
    n, m = a.shape
    if n != m:
        raise ValueError(f"Matrix A is not square (A.shape = {a.shape})")

    # Validate right hand side matrix shape
    if 1 not in b.shape:
        raise ValueError(f"b Matrix must be a vector (Got b.shape = {b.shape})")
    b = b if b.cols == 1 else b.T  # Ensure b is a column matrix
    if b.rows != n:
        raise ValueError(f"The number of rows of A and b do not match (A.shape = {a.shape}, b.shape = {b.shape})")
    return a, b


def gauss_naive(a: sympy.Matrix, b: sympy.Matrix, verbose=False) -> sympy.Matrix:
    """
    :param verbose: display intermediate steps in Jupyter Notebook
    :param a: Coefficient matrix
    :param b: right hand side
    :return: solution vector
    """
    a, b = validate_ax_b_shapes(a=a, b=b)

    # Create augmented matrix
    aug: sympy.Matrix = a.row_join(b)

    # ======== Forward Elimination ========
    if verbose: display(Markdown(f"**Forward Elimination**"))
    aug = forward_elimination(aug, verbose=verbose)

    # ======== Back Substitution ========
    if verbose: display(Markdown(f"**Back Substitution**"))
    x = back_substitution(aug)

    if verbose:
        #display results
        display(Math(numpy_to_latex_gauss(aug)))
        display(Markdown(f"**Solution Vector**"))
        display(Math(
            f"{vector2latex({f'x_{{{i}}}': j for i, j in enumerate(x.tolist(), start=1)}, brackets='[]')}"
            f" = "
            f"{sympy.latex(x.evalf(10))}"
        ))
    return x


def lu_decomposition_linear_solver(
        a: sympy.Matrix,
        b: sympy.Matrix,
        # decomposition: str,
        verbose=False
) -> sympy.Matrix:
    """

    :param a:
    :param b:
    # :param decomposition:
    :param verbose:
    :return:
    """
    a, b = validate_ax_b_shapes(a=a, b=b)

    # Step 1: Decomposition
    l, u, _ = a.LUdecomposition()
    x_sym_matrix = sympy.Matrix(sympy.symbols(f"x_{{1:{a.rows+1}}}"))
    if verbose:
        display(Markdown(f"**LU Decomposition**"))
        display(Math(
            f'{sympy.latex(a)}'
            f'{sympy.latex(x_sym_matrix)} = '
            f'{sympy.latex(b)}'
        ))
        display(Math(
            f'{sympy.latex(l)}'
            f'{sympy.latex(u)}'
            f'{sympy.latex(x_sym_matrix)} = '
            f'{sympy.latex(b)}'
        ))

    # Step 2: Solve Ly=b using forward substitution
    ly = l.row_join(b)
    y = forward_substitution(ly)
    y_sym_matrix = sympy.Matrix(sympy.symbols(f"y_{{1:{a.rows+1}}}"))

    if verbose:
        display(Markdown(f"**$Ly=b$ forward substitution**"))
        display(Math(
            f'{sympy.latex(l)}'
            f'{sympy.latex(y_sym_matrix)} = '
            f'{sympy.latex(b)}'
        ))
        display(Math(
            f'{sympy.latex(y_sym_matrix)} = {sympy.latex(y)}'
        ))

    # Step 3: Solve Ux=y
    ux = u.row_join(y)
    x = back_substitution(ux)

    if verbose:
        display(Markdown(f"**$Lx=y$ back substitution**"))
        display(Math(
            f'{sympy.latex(l)}'
            f'{sympy.latex(x_sym_matrix)} = '
            f'{sympy.latex(y)}'
        ))

        # display results
        display(Markdown(f"**Solution Vector**"))
        display(Math(
            f'{sympy.latex(x_sym_matrix)} = {sympy.latex(x)} ='
            f'{sympy.latex(x.evalf(6))}'
        ))

    return x






