from dataclasses import dataclass, field
from typing import Dict, Any, ClassVar, Set

import numpy as np
import sympy

from Numerical import Numerical


@dataclass
class EulerMethod(Numerical):
    function: sympy.Function = field(default=None)
    x0: float = field(default=0)
    dx: float = field(default=0.1)
    variables: ClassVar[Set[str]] = {'x', 'slope', 'dx'}

    def y(self, x: float) -> float:
        return self.function.subs(x=x)

    def _validate_initial_state(self) -> None:
        if not self.function:
            raise ValueError('Function not provided')
        elif not isinstance(self.function, sympy.Function):
            raise ValueError(f'Function must be a SymPy function. Got: {type(self.function)}')


    def initialize(self) -> None:
        dy = self.y(self.x0 + self.dx) - self.y(self.x0)
        slope_0 = dy / self.dx

        self.initial_state = dict(
            x=self.x0,
            slope=slope_0,
            dx=self.dx
        )
        super().initialize()

    def step(self) -> Dict[str, Any]:
        """
        New value = old value + slope × step size
        :return: New value
        """
        xn = self.history['x'][self.last_iteration]
        dy = self.y(xn + self.dx) - self.y(xn)
        slope = dy / self.dx
        dx = self.dx
        return dict(
            x=xn + slope * dx,
            slope=slope,
            dx=dx
        )
