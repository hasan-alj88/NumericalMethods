from abc import ABC
from dataclasses import dataclass, field
from typing import List

import pandas as pd

from Numerical import Numerical


@dataclass
class RootFinder(Numerical, ABC):
    function: callable = field(default=None)
    roots: List[float] = field(default_factory=list, init=False)

    def _validate_initial_state(self) -> None:
        self._single_argument_function(self.function)


    def error_analysis(self, exact_solution: List[float]) -> pd.DataFrame:
        df = self.history.copy()
        root_names = ['x_root'] + [f'x_root_{i}' for i in range(1, len(self.roots) + 1)]
        for name, root in zip(root_names, self.roots):
            df[f'{name}_absolute_error'] = self.absolute_error(df[name], exact_solution)
            df[f'{name}_relative_error'] = self.relative_error(df[name], exact_solution)
        return df

