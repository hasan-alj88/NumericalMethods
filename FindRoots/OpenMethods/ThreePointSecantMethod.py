from dataclasses import dataclass, field

from FindRoots.RootFinder import RootFinder
from StopConditions.StopIfEqual import StopIfZero
from StopConditions.StopIfNaN import StopIfNaN


@dataclass
class ThreePointSecantMethod(RootFinder):
    x0: float = field(default=0.0)
    dx: float = field(default=None)
    independent_variable_count: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if self.x0 is None:
            raise ValueError("x0 cannot be None")
        if self.dx is None:
            raise ValueError("dx cannot be None")

        self.add_stop_condition(StopIfZero(tracking='f', patience=self.patience,
                                           absolute_tolerance=self.absolute_tolerance,
                                           relative_tolerance=self.relative_tolerance))
        self.add_stop_condition(StopIfNaN(track_variables=['f', 'x', 'df_dx', 'd2f_dx2']))

    def df(self, x) -> float:
        x0 = x - self.dx
        x1 = x + self.dx
        return (self.function(x1) - self.function(x0)) / (2 * self.dx)

    def d2f(self, x) -> float:
        x0 = x - self.dx
        x1 = x
        x2 = x + self.dx
        f0 = self.function(x0)
        f1 = self.function(x1)
        f2 = self.function(x2)
        return (f2 - 2 * f1 + f0) / self.dx ** 2

    @property
    def initial_state(self) -> dict:
        return dict(
            x=self.x0,
            f=self.function(self.x0),
            df_dx=self.df(self.x0),
            d2f_dx2=self.d2f(self.x0),
            det =self.df(self.x0) * self.d2f(self.x0),
            log='Initial State'
        )

    def step(self) -> dict:
        x_n = self.history['x']
        f_n = self.history['f']

        df_dx = self.df(x_n)
        d2f_dx2 = self.d2f(x_n)

        det = df_dx ** 2 - 2 * f_n * d2f_dx2

        x_np1_1 = np.real(x_n - (df_dx-np.sqrt(det)) / d2f_dx2)
        x_np1_2 = np.real(x_n - (df_dx+np.sqrt(det)) / d2f_dx2)
        f_np1_1 = self.function(x_np1_1)
        f_np1_2 = self.function(x_np1_2)
        if abs(f_np1_1) < abs(f_np1_2):
            x_np1 = x_np1_1
            f_np1 = f_np1_1
            log = f'det > 0, x_np1_1 is better'
        else:
            x_np1 = x_np1_2
            f_np1 = f_np1_2
            log = f'det > 0, x_np1_2 is better'


        return dict(
            x=x_np1,
            f=f_np1,
            df_dt=df_dx,
            d2f_dt2=d2f_dx2,
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
        x0=0,
        dx=0.01,
        max_iterations=100
    )
    df = solver.run()
    print(df)
