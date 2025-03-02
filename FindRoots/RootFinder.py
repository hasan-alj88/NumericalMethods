from abc import ABC
from dataclasses import dataclass, field
from typing import List

import pandas as pd

from Core.Numerical import Numerical
from utils.ErrorCalculations import absolute_error, relative_error
from utils.ValidationTools import function_arg_count


@dataclass
class RootFinder(Numerical, ABC):
    function: callable = field(default=None)
    independent_variable_count: int = 1

    def _validate_initial_state(self) -> None:
        if function_arg_count(self.function) == self.independent_variable_count:
            raise ValueError(f"Number of independent variables {function_arg_count(self.function)} while"
                             f" {self.function} has {self.independent_variable_count}")

    def error_analysis(self, exact_solution:float) -> pd.DataFrame:
        df = self.history.to_data_frame.copy()
        df[f'absolute_error'] = absolute_error(df['t'], exact_solution)
        df[f'relative_error'] = relative_error(df['t'], exact_solution)
        return df

