from dataclasses import dataclass, field

import numpy as np

from FindRoots.RootFinder import RootFinder
from StopConditions.StopIfEqual import StopIfZero
from StopConditions.StopIfNaN import StopIfNaN


@dataclass
class ThreePointSecantMethod(RootFinder):
    t0: float = field(default=0.0)
    dt: float = field(default=None)
    independent_variable_count: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if self.t0 is None:
            raise ValueError("t0 cannot be None")
        if self.dt is None:
            raise ValueError("dt cannot be None")

        self.add_stop_condition(StopIfZero(tracking='f', patience=self.patience,
                                           absolute_tolerance=self.absolute_tolerance,
                                           relative_tolerance=self.relative_tolerance))
        self.add_stop_condition(StopIfNaN(track_variables=['f', 't', 'df_dt', 'd2f_dt2']))

    def df(self, t) -> float:
        t0 = t - self.dt
        t1 = t + self.dt
        return (self.function(t1) - self.function(t0)) / (2*self.dt)

    def d2f(self, t) -> float:
        t0 = t - self.dt
        t1 = t
        t2 = t + self.dt
        f0 = self.function(t0)
        f1 = self.function(t1)
        f2 = self.function(t2)
        return (f2 - 2 * f1 + f0) / self.dt ** 2

    @property
    def initial_state(self) -> dict:
        return dict(
            t=self.t0,
            f=self.function(self.t0),
            df_dt=self.df(self.t0),
            d2f_dt2=self.d2f(self.t0),
            det =self.df(self.t0) * self.d2f(self.t0),
            log='Initial State'
        )

    def step(self) -> dict:
        t_n = self.history['t']
        f_n = self.history['f']

        df_dt = self.df(t_n)
        d2f_dt2 = self.d2f(t_n)

        det = df_dt ** 2 - 2 * f_n * d2f_dt2

        # if det > 0:
        #     t_np1_1 = t_n - (df_dt-np.sqrt(det)) / d2f_dt2
        #     t_np1_2 = t_n - (df_dt+np.sqrt(det)) / d2f_dt2
        #     f_np1_1 = self.function(t_np1_1)
        #     f_np1_2 = self.function(t_np1_2)
        #     if abs(f_np1_1) < abs(f_np1_2):
        #         t_np1 = t_np1_1
        #         f_np1 = f_np1_1
        #         log = f'det > 0, t_np1_1 is better'
        #     else:
        #         t_np1 = t_np1_2
        #         f_np1 = f_np1_2
        #         log = f'det > 0, t_np1_2 is better'
        # elif det < 0:
        #     t_np1 = t_n - f_n/ df_dt
        #     f_np1 = self.function(t_np1)
        #     log = f'det < 0, using newton-raphson'
        # else:
        #     t_np1 = t_n - df_dt / d2f_dt2
        #     f_np1 = self.function(t_np1)
        #     log = f'det = 0'

        t_np1_1 = np.real(t_n - (df_dt-np.sqrt(det)) / d2f_dt2)
        t_np1_2 = np.real(t_n - (df_dt+np.sqrt(det)) / d2f_dt2)
        f_np1_1 = self.function(t_np1_1)
        f_np1_2 = self.function(t_np1_2)
        if abs(f_np1_1) < abs(f_np1_2):
            t_np1 = t_np1_1
            f_np1 = f_np1_1
            log = f'det > 0, t_np1_1 is better'
        else:
            t_np1 = t_np1_2
            f_np1 = f_np1_2
            log = f'det > 0, t_np1_2 is better'


        return dict(
            t=t_np1,
            f=f_np1,
            df_dt=df_dt,
            d2f_dt2=d2f_dt2,
            det=det,
            log=log,
        )

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    solver = ThreePointSecantMethod(
        function=lambda t: -np.tanh(t**2-9),
        t0=0,
        dt=0.01,
        max_iterations=100
    )
    df = solver.run()
    print(df)
