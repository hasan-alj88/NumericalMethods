from abc import ABC
from dataclasses import dataclass, field
from typing import Callable, Set

import pandas as pd

from Core import Numerical
from StopConditions.StopIfEqual import StopIfZero
from utils.ErrorCalculations import absolute_error, relative_error
from utils.ValidationTools import function_arg_count


@dataclass
class FirstOrderODE(Numerical, ABC):
    """
    Abstract base class for numerical methods for solving first-order ODEs.
    dx/dt = f(x(t), t)
    - first-order: height derivative is 1.
    """
    derivative_function: Callable = field(default=None)
    t0: float = field(default=0)
    x0: float = field(default=0)
    dt: float = field(default=0.1)

    def __post_init__(self):
        self._validate_initial_state()
        super().__post_init__()


    def _validate_initial_state(self) -> None:
        if function_arg_count(self.derivative_function) != 2:
            raise ValueError(f'Derivative function must take 2 arguments f(x,t), '
                             f'not {function_arg_count(self.derivative_function)}')

        if self.x0 is None:
            raise ValueError('Initial state must include x0')
        if self.dt is None:
            raise ValueError('Initial state must include dt')
        if self.t0 is None:
            raise ValueError('Initial state must include t0')

    def error_analysis(self, analytic_solution_function: callable) -> pd.DataFrame:
        df = self.history.to_data_frame.copy()
        exact_solution = df.apply(lambda row: analytic_solution_function(row['t']), axis=1)
        df['$|x_{n} - x_{exact}|$'] = absolute_error(df['x'], exact_solution)
        df[r'$\frac{|x_{n} - x_{exact}|}{x_{exact}}$'] = relative_error(df['x'], exact_solution)
        return df
