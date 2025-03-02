from dataclasses import dataclass, field

from FindRoots.RootFinder import RootFinder
from StopConditions.StopIfEqual import StopIfZero


@dataclass
class SecantMethod(RootFinder):
    t0: float = field(default=0.0)
    t1: float = field(default=None)
    independent_variable_count: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if self.t0 is None:
            raise ValueError("t0 cannot be None")
        if self.t1 is None:
            raise ValueError("t1 cannot be None")

        self.add_stop_condition(StopIfZero(tracking='f', patience=3, tolerance=1e-6))
        super().__post_init__()

    @property
    def initial_state(self) -> dict:
        return dict(t=self.t1, t_mn1=self.t0, f=self.function(self.t1), f_mn1=self.function(self.t1))


    def step(self) -> dict:
        """
        t_{n+1} = t_{n} - \frac{f(t_n)(t_{n-1}-x_n)}{f(t_{n-1})-f(t_{n})}
        """
        t_nm1 = self.history['t_mn1']
        t_n = self.history['t']

        f_nm1 = self.function(t_nm1)
        f_n = self.function(t_n)

        t_np1 = t_n - f_n * (t_nm1 - t_n) / (f_n - f_nm1)

        return dict(
            t=t_np1,
            t_mn1=t_n,
            f=self.function(t_np1),
            f_mn1=f_n,
        )