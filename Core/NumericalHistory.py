import logging
from dataclasses import field, dataclass
from typing import List, Any, Optional, Set

import pandas as pd
from matplotlib import pyplot as plt

from utils.LaTeXTools import df_to_latex
from utils.log_config import get_logger


@dataclass
class NumericalHistory:
    parameters: Set[str] = field(default_factory=list)
    data: List[dict] = field(default_factory=list)
    console_log_level: int|str = field(default='OFF')
    _logger: logging.Logger = field(default=None, init=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item) -> Any:
        """Get the last state of an item"""
        return self.data[-1][item] if self.data else None

    def __call__(self, iteration: int, item: str) -> Any:
        """Get the state of an item at a given iteration"""
        return self.data[iteration][item] if self.data else None

    @property
    def logger(self):
        if self._logger is not None:
            return self._logger
        self._logger =get_logger(self.__class__.__name__, self.console_log_level)
        return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = value

    @property
    def last_iteration(self) -> int:
        return len(self.data) - 1 if self.data else -1

    @property
    def last_state(self) -> dict:
        return self.data[-1] if self.data else None

    def record_state(self, state: dict) -> None:
        self.data.append(state)

    @property
    def to_data_frame(self) -> pd.DataFrame:
        return pd.DataFrame(data=[iteration for iteration in self.data])

    def to_latex(
            self,
            filepath: str,
            variables: List[str] = None,
            iterations: List[int] = None,
            precision: int = 4,
            caption: str = None,
            label: str = None,
            formatting: dict = None,
            logger = get_logger(__name__)
    ):
        df_to_latex(
            df=self.to_data_frame,
            filepath=filepath,
            variables=variables or [],
            iterations=iterations,
            precision=precision,
            caption=caption,
            label=label,
            formatting=formatting,
            logger = logger
        )

    def to_json(self, filepath: str):
        self.to_data_frame.to_json(filepath, orient='records')

    def to_csv(self, filepath: str):
        self.to_data_frame.to_csv(filepath, index=False)

    def plot(self,
             x_var: Optional[str],
             y_var: str,
             ax: plt.Axes,
             *args, **kwargs) -> plt.Axes:
        history = self.to_data_frame
        if x_var is not None and x_var not in history.columns:
            raise ValueError(f"Variable '{x_var}' not found in history. "
                             f"Available variables: {list(history.columns)}")
        if y_var not in history.columns:
            raise ValueError(f"Variable '{y_var}' not found in history. "
                             f"Available variables: {list(history.columns)}")

        xdata = history[x_var] if x_var is not None else history.index
        ydata = history[y_var]
        ax.plot(xdata, ydata, *args, **kwargs)
        return ax




