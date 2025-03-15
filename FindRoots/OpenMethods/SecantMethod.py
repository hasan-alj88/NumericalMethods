from dataclasses import dataclass, field

import numpy as np

from FindRoots.RootFinder import RootFinder
from StopConditions.StopIfEqual import StopIfZero
from StopConditions.StopIfNaN import StopIfNaN


@dataclass
class SecantMethod(RootFinder):
    function: callable = field(default=None)
    xm1: float = field(default=0.0)
    x0: float = field(default=None)
    independent_variable_count: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if self.xm1 is None:
            raise ValueError("x_{-1} cannot be None")
        if self.x0 is None:
            raise ValueError("x_0 cannot be None")

        self.add_stop_condition(StopIfZero(tracking='f', patience=self.patience,
                                           absolute_tolerance=self.absolute_tolerance,
                                           relative_tolerance=self.relative_tolerance))
        self.add_stop_condition(StopIfNaN(track_variables=['f','x', 'x_nm1','df_dx','d2f_dx2']))



    @property
    def initial_state(self) -> dict:
        return dict(
            x=self.x0,
            x_nm1=self.xm1,
            f=self.function(self.x0),
            f_nm1=self.function(self.x0),
            df_dt=(self.function(self.x0) - self.function(self.xm1)) / (self.x0 - self.xm1)
        )


    def step(self) -> dict:
        """
        x_{n+1} = x_{n} - \frac{f(x_n)(x_{n-1}-t_n)}{f(x_{n-1})-f(x_{n})}
        """
        x_nm1 = self.history['t_nm1']
        x_n = self.history['t']

        f_nm1 = self.function(x_nm1)
        f_n = self.function(x_n)

        df_dx = (f_n - f_nm1) / (x_n - x_nm1)
        df_dx = df_dx if abs(df_dx) != 0 else np.nan
        x_np1 = x_n - f_n / df_dx

        return dict(
            x=x_np1,
            x_nm1=x_n,
            f=self.function(x_np1),
            f_nm1=f_n,
            dx_dt=df_dx,
        )