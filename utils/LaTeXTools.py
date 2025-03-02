import pandas as pd
from typing import List

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
        "\\begin{table}[htbp]", # noqa
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