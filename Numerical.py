import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Generator, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from log_config import get_logger

logger = get_logger(__name__)


@dataclass
class StopCondition(ABC):
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    stop_reason: str = field(default='', init=False)
    _generator: Optional[Generator[Tuple[bool, str], None, None]] = field(default=None, init=False)

    @property
    def last_iteration(self) -> int:
        return self.history.index.max() if not self.history.empty else 0

    @abstractmethod
    def stop_condition_generator(self) -> Generator[Tuple[bool, str], None, None]:
        """Create a generator that will yield stop condition results"""
        pass

    def _initialize_generator(self) -> None:
        """Initialize the generator"""
        if self._generator is None:
            self._generator = self.stop_condition_generator()

    def next(self, history: pd.DataFrame) -> Tuple[bool, str]:
        """
        Executes the next step in the generation process, managing the state of the generator
        and responding to halt conditions.

        :param history: The current state DataFrame with iterations as index and variables as columns
        :type history: pd.DataFrame
        :return: A tuple where the first element is a boolean indicating whether the
            generator should stop, and the second element is the reason for stopping.
        :rtype: Tuple[bool, str]
        """
        while True:
            self._initialize_generator()
            try:
                self.history = history
                should_stop, reason = next(self._generator)
                return should_stop, reason
            except StopIteration:
                return True, f'Stop condition [{self.__class__.__name__}] met: {self.stop_reason}'


@dataclass
class Numerical(ABC):
    initial_state: dict = field(init=False)
    stop_conditions: List[StopCondition] = field(default_factory=list)
    max_iterations: int = 1_000
    verbose: bool = False
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    _iteration: int = field(default=0, init=False)

    def add_stop_condition(self, stop_condition: StopCondition) -> None:
        """Add a stop condition to the list of stop conditions"""
        self.stop_conditions.append(stop_condition)

    @abstractmethod
    def initialize(self) -> None:
        self.history = pd.DataFrame()
        self._iteration = 0
        self.record_state(self.initial_state)

    @abstractmethod
    def _validate_initial_state(self) -> None:
        pass

    @abstractmethod
    def step(self) -> dict:
        pass

    def record_state(self, state: dict) -> None:
        new_row = pd.DataFrame([state], index=[self._iteration])
        self.history = pd.concat([self.history, new_row], verify_integrity=True)

    def _check_stop_conditions(self) -> Generator[str, None, None]:
        """
        Checks if any of the specified stopping conditions are met
        :yields: Always yields True until a stop condition is met
        """
        for iteration in range(1, self.max_iterations+1):
            self._iteration = iteration
            if iteration == 0:
                yield f'Initial Iteration completed'
                continue

            met_stop_conditions = []
            unmet_stop_conditions = []
            for stop_cond in self.stop_conditions:
                stop_cond_met, reason = stop_cond.next(self.history)
                condition_name = stop_cond.__class__.__name__
                if stop_cond_met:
                    met_stop_conditions.append(f"Stop condition [{condition_name:<15}] met    : {reason}")
                else:
                    unmet_stop_conditions.append(f"Stop condition [{condition_name:<15}] not met: {reason}")

            status = '\n'.join(met_stop_conditions + unmet_stop_conditions)
            should_stop = len(unmet_stop_conditions) != len(self.stop_conditions)
            if should_stop:
                yield status
                break
            yield f'Iteration {iteration} completed\n{status}'

        else:
            logger.info(f"Stop condition max iterations reached ({self.max_iterations})")

    def run(self) -> pd.DataFrame:
        """
        Run the numerical method using a for-loop with the generator method
        that checks stop conditions using StopIteration.
        :returns: pd.DataFrame: Complete history of the computation
        """
        self.history = pd.DataFrame()

        logger.info(f"Starting {self.__class__.__name__}")
        self.initialize()

        for status in self._check_stop_conditions():
            logger.info(f"{status}")
            logger.info(f"Iteration {self._iteration}....")
            state = self.step()
            self.record_state(state)
            logger.info(f"State: \n{pd.Series(state).to_string()}\n")

        return self.history

    @property
    def iteration(self) -> int:
        """Current iteration number (read-only)"""
        return self._iteration

    @property
    def last_iteration(self) -> int:
        """Maximum number of iterations"""
        return self.history.index.max() if not self.history.empty else -1

    def get_history_at(self, iteration: int) -> dict:
        """
        Get all variables' values at a specific iteration.
        :param iteration: Iteration number
        :return: dict: Variable values at specified iteration
        """
        if iteration < 0:
            raise ValueError(f"Iteration must be non-negative, got {iteration}")
        if iteration > self.last_iteration:
            raise ValueError(f"Iteration {iteration} exceeds maximum recorded iteration {self.last_iteration}")

        return self.history.loc[iteration].to_dict()

    def __getitem__(self, item):
        """DataFrame-style access to variable history"""
        if item not in self.history.columns:
            raise KeyError(f"Variable '{item}' not found in history")
        return self.history[item]

    def get_last_state(self) -> dict:
        """Get the last state of the numerical method"""
        return self.get_history_at(self.last_iteration)

    def export_history(self, filepath: str) -> None:
        """
        Export history to a JSON file

        Args:
            filepath: Path to save the history
        """
        self.history.to_json(filepath, indent=2, orient='split')

    def plot_history(self,
                     x_var: Optional[str],
                     y_var: str,
                     ax: plt.axis,
                     *args, **kwargs) -> plt.axis:
        """
        Plot history of a variable vs iterations.
        :param x_var: x-axis variable name.
        :param y_var: y-axis variable name.
        :param ax: Plot axis
        """
        if x_var is not None and x_var not in self.history.columns:
            raise ValueError(f"Variable '{x_var}' not found in history. "
                             f"Available variables: {list(self.history.columns)}")
        if y_var not in self.history.columns:
            raise ValueError(f"Variable '{y_var}' not found in history. "
                             f"Available variables: {list(self.history.columns)}")

        xdata = self.history[x_var] if x_var is not None else self.history.index
        ydata = self.history[y_var]
        ax.plot(xdata, ydata, *args, **kwargs)
        return ax

    def export_to_latex(self, filepath: str, variables: List[str] = None,
                        iterations: List[int] = None, precision: int = 4,
                        caption: str = None, label: str = None,
                        formatting: dict = None) -> None:
        """
        Export history to a LaTeX table

        Args:
            filepath: Path to save the LaTeX table
            variables: List of variable names to include (None for all variables)
            iterations: List of iterations to include (None for all iterations)
            precision: Number of decimal places for floating point values
            caption: Optional caption for the table
            label: Optional label for the table
            formatting: Optional dictionary with formatting options for variables
        """
        df_to_latex(
            self.history,
            filepath,
            variables=variables,
            iterations=iterations,
            precision=precision,
            caption=caption,
            label=label,
            formatting=formatting
        )

    @staticmethod
    def absolute_error(x, y):
        return np.abs(x - y)

    @staticmethod
    def relative_error(x, y):
        return np.abs((x - y) / x)


def df_to_latex(
        df: pd.DataFrame,
        filepath: str,
        variables: List[str] = None,
        iterations: List[int] = None,
        precision: int = 4,
        caption: str = None,
        label: str = None,
        formatting: dict = None
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
        "\\begin{table}[htbp]",
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