from dataclasses import dataclass, field

import numpy as np

from FindRoots.RootFinder import RootFinder
from StopConditions.StopIfEqual import StopIfZero
from StopConditions.StopIfNaN import StopIfNaN


@dataclass
class SecantMethod(RootFinder):
    function: callable = field(default=None)
    t0: float = field(default=0.0)
    t1: float = field(default=None)
    independent_variable_count: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if self.t0 is None:
            raise ValueError("t0 cannot be None")
        if self.t1 is None:
            raise ValueError("t1 cannot be None")

        self.add_stop_condition(StopIfZero(tracking='f', patience=self.patience,
                                           absolute_tolerance=self.absolute_tolerance,
                                           relative_tolerance=self.relative_tolerance))
        self.add_stop_condition(StopIfNaN(track_variables=['f','t', 't_nm1','df_dt','d2f_dt2']))



    @property
    def initial_state(self) -> dict:
        return dict(
            t=self.t1,
            t_nm1=self.t0,
            f=self.function(self.t1),
            f_nm1=self.function(self.t1),
            df_dt=(self.function(self.t1) - self.function(self.t0)) / (self.t1 - self.t0)
        )


    def step(self) -> dict:
        """
        t_{n+1} = t_{n} - \frac{f(t_n)(t_{n-1}-t_n)}{f(t_{n-1})-f(t_{n})}
        """
        t_nm1 = self.history['t_nm1']
        t_n = self.history['t']

        f_nm1 = self.function(t_nm1)
        f_n = self.function(t_n)

        df_dt = (f_n - f_nm1) / (t_n - t_nm1)
        df_dt = df_dt if abs(df_dt) != 0 else np.nan
        t_np1 = t_n - f_n / df_dt

        return dict(
            t=t_np1,
            t_nm1=t_n,
            f=self.function(t_np1),
            f_nm1=f_n,
            df_dt=df_dt,
        )