from abc import ABC
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from Core import Numerical
from StopConditions.StopIfGreaterThan import StopIfGreaterThan
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
    t_final: float = field(default=None)

    def __post_init__(self):
        if function_arg_count(self.derivative_function) != 2:
            raise ValueError(f'Derivative function must take 2 arguments f(x,t), '
                             f'not {function_arg_count(self.derivative_function)}')

        if self.x0 is None:
            raise ValueError('Initial state must include x0')
        if self.dt is None:
            raise ValueError('Initial state must include dt')
        if self.t0 is None:
            raise ValueError('Initial state must include t0')
        if self.t_final is None:
            raise ValueError('Initial state must include t_final')
        elif self.t_final <= self.t0:
            raise ValueError(f't_final({self.t_final}) must be greater than t0 ({self.t0})')
        self.add_stop_condition(StopIfGreaterThan(tracking='t', threshold=self.t_final-2*self.dt))


    def error_analysis(self, analytic_solution_function: callable) -> pd.DataFrame:
        df = self.history.to_data_frame.copy()
        exact_solution = df.apply(lambda row: analytic_solution_function(row['t']), axis=1)
        df['\varepsilon'] = absolute_error(df['x'], exact_solution)
        df[r'\varepsilon_r'] = relative_error(df['x'], exact_solution)
        return df
