from dataclasses import dataclass, field
from typing import Dict, Any

from FindRoots.RootFinder import RootFinder
from StopConditions.StopIfNaN import StopIfNaN


@dataclass
class ModifiedSecantMethod(RootFinder):
    function: callable = field(default=None)
    independent_variable_count: int = 1
    t0: float = field(default=0.0)
    dt: float = field(default=None)

    def __post_init__(self):
        if self.dt is None:
            raise ValueError("dt cannot be None")
        if self.t0 is None:
            raise ValueError("t0 cannot be None")
        self.add_stop_condition(StopIfNaN(track_variables=['f', 't']))

    @property
    def initial_state(self) -> dict:
        return dict(
            t=self.t0,
            f=self.function(self.t0)
        )

    def step(self) -> Dict[str, Any]:
        """
        t_{n+1} = t_n -\frac{dt * f(t_n)}{f(t_n) - f(t_n-dt)}}
        :return:
        """

        t_n = self.history['t']
        f_n = self.history['f']

        t_np1 = t_n - self.dt * f_n / (f_n - self.function(t_n - self.dt))
        f_np1 = self.function(t_np1)

        return dict(
            t=t_np1,
            f=f_np1
        )