import numpy as np
import pandas as pd
from typing import List

import sympy
from IPython.core.display import Math
from IPython.core.display_functions import display

from utils.log_config import get_logger


def df_to_latex(
        df: pd.DataFrame,
        filepath: str,
        variables: List[str] = None,
        iterations: List[int] = None,
        precision: int = 4,
        caption: str = None,
        label: str = None,
        formatting: dict = None,
        logger = get_logger(__name__)
) -> str:
    """
    Convert DataFrame to a LaTeX table and save to file

    Args:
        df: DataFrame with iterations as index and variables as columns
        filepath: Path to save the LaTeX table
        variables: List of variable names to include (None for all variables)
        iterations: List of iterations to include (None for all iterations)
        precision: Number of decimal places for floating point values
        caption: Optional caption for the table
        label: Optional label for the table
        formatting: Optional dictionary with formatting options for variables
        logger: Optional logger instance (defaults to root logger)
    Returns:
        str: The complete LaTeX table content
    """
    # Use all variables if none specified
    if variables is None:
        variables = list(df.columns)

    # Filter DataFrame
    df_subset = df[variables].copy()

    if iterations is not None:
        df_subset = df_subset.loc[iterations]

    # Apply formatting
    if formatting is None:
        formatting = {}

    new_columns = []
    for var in variables:
        if var in formatting and 'format' in formatting[var]:
            format_spec = formatting[var]['format']
            if isinstance(format_spec, str):
                df_subset[var] = df_subset[var].apply(lambda x: f'{x:{format_spec}}')
            elif callable(format_spec):
                df_subset[var] = df_subset[var].apply(format_spec)

        new_header = formatting.get(var, {}).get('header', f"{var}")
        new_columns.append(new_header)

    df_subset.columns = new_columns

    # Convert to LaTeX
    latex_lines = [
        "\\begin{table}[h]", # noqa
        "\\centering",
        df_subset.to_latex(
            float_format=lambda x: f'{x:.{precision}f}',
            escape=False,
            index=True,
            index_names=True
        ),
    ]

    if caption:
        latex_lines.append(f"\\caption{{{caption}}}")
    if label:
        latex_lines.append(f"\\label{{{label}}}")

    latex_lines.append("\\end{table}")
    latex_content = '\n'.join(latex_lines)

    with open(filepath, 'w') as file:
        file.write(latex_content)

    logger.info(f"LaTeX table exported to {filepath}")
    return latex_content

def numpy_to_latex_gauss(aug: np.ndarray | sympy.Matrix) -> str:

    if isinstance(aug, sympy.Matrix):
        aug = np.array(aug.tolist())
    elif not isinstance(aug, np.ndarray):
        try:
            aug = np.array(aug)
        except TypeError:
            raise TypeError(f'Provided array is not a numpy array, Got {type(aug)}')

    n,c = aug.shape
    matrix_latex = "\n\\left[\\begin{array}{" + "c" * (c-1) + "|c}" + "\n"

    # Convert each row to a LaTeX row
    for row in aug:
        matrix_latex += "  " + " & ".join(map(str, row)) + " \\\\ \n"

    matrix_latex += "\\end{array}\\right]\n"
    return matrix_latex

def default_format_function(x)->str:
    if isinstance(x, (float, np.floating)):
        return f'{x:,.3f}'
    else:
        return str(x)


def numpy2latex(
        a,
        format_function:callable=default_format_function,
        brackets: str = "") -> str:
    """Convert a numpy array to LaTeX code."""
    env = {
        '[]': 'bmatrix',
        '{}': 'Bmatrix',
        '()': 'pmatrix',
        '||': 'vmatrix',
        '|': 'Vmatrix',
        '': 'array',
    }
    env_selection = env.get(brackets, None)
    if env_selection is None:
        raise ValueError(f"Invalid brackets value: {brackets}. Valid options are {list(env.keys())}.")

    if not isinstance(a, np.ndarray):
        try:
            a = np.array(a)
        except Exception as e:
            raise ValueError(f"Input object `a` cannot be converted to a numpy array: {e}")

    if a.size == 0:
        raise ValueError("Array `a` is empty. Cannot generate LaTeX code for an empty array.")
    if a.ndim == 1:
        a = a[:, np.newaxis]
    elif a.ndim > 2:
        raise ValueError("Array must be 1 or 2 dimensions")

    open_env = r'\begin{' + f'{env_selection}' + r'}'
    if env_selection == "array":
        open_env += '{' + 'c' * a.shape[1] + '}'
    close_env = r'\end{' + f'{env_selection}' + r'}'
    lines = [open_env]
    for row in a:
        line = ' & '.join([format_function(value) for value in row]) + r' \\'
        lines.append(line)
    lines.append(close_env)
    return '\n'.join(lines)

def vector2latex(vector_dict, brackets='{}',
                 lhs_format_function:callable=default_format_function,
                 rhs_format_function:callable=default_format_function
                 )->str:
    lhs = vector_dict.keys()
    rhs = vector_dict.values()
    lhs_str = numpy2latex(np.array(list(lhs)), brackets=brackets, format_function=lhs_format_function)
    rhs_str = numpy2latex(np.array(list(rhs)), brackets=brackets, format_function=rhs_format_function)
    return f'{lhs_str} = {rhs_str}'