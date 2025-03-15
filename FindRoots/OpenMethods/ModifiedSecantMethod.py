from dataclasses import dataclass, field
from typing import Dict, Any

from FindRoots.RootFinder import RootFinder
from StopConditions.StopIfNaN import StopIfNaN


@dataclass
class ModifiedSecantMethod(RootFinder):
    function: callable = field(default=None)
    independent_variable_count: int = 1
    x0: float = field(default=0.0)
    h: float = field(default=None)

    def __post_init__(self):
        if self.h is None:
            raise ValueError("h cannot be None")
        if self.x0 is None:
            raise ValueError("x0 cannot be None")
        self.add_stop_condition(StopIfNaN(track_variables=['f', 't']))

    @property
    def initial_state(self) -> dict:
        return dict(
            x=self.x0,
            f=self.function(self.x0)
        )

    def step(self) -> Dict[str, Any]:
        """
        x_{n+1} = x_n -\frac{h * f(x_n)}{f(x_n) - f(x_n-h)}}
        :return:
        """

        x_n = self.history['x']
        f_n = self.history['f']

        x_np1 = x_n - self.h * f_n / (f_n - self.function(x_n - self.h))
        f_np1 = self.function(x_np1)

        return dict(
            x=x_np1,
            f=f_np1
        )