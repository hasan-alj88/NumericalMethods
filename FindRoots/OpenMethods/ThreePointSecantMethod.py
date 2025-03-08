from dataclasses import dataclass, field

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
        return (self.function(t1) - self.function(t0)) / (t1 - t0)

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
        )

    def step(self) -> dict:
        """
        t_{n+1} = t_n - \frac{f(t_n)}{f'(t_n)} \left( 1 + \frac{1}{2} \frac{f(t_n) f''(t_n)}{(f'(t_n))^2} \right).
        """
        t_n = self.history['t']
        f_n = self.history['f']

        df_dt = self.df(t_n)
        d2f_dt2 = self.d2f(t_n)

        t_np1 = t_n - (f_n / df_dt) * (1 + 0.5 * f_n * d2f_dt2 / df_dt ** 2)
        f_np1 = self.function(t_np1)

        return dict(
            t=t_np1,
            f=f_np1,
            df_dt=df_dt,
            d2f_dt2=d2f_dt2,
        )