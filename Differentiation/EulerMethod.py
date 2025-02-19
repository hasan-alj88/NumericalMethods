from dataclasses import dataclass, field
from typing import Dict, Any, ClassVar, Set, Callable

import sympy

from Numerical import Numerical


@dataclass
class EulerMethod(Numerical):
    function: Callable = field(default=None)
    x0: float = field(default=0)
    dx: float = field(default=0.1)
    variables: ClassVar[Set[str]] = {'x', 'slope', 'dx'}

    def __post_init__(self):
        if self.function is None:
            raise ValueError('Function cannot be provided at initialization')
        if not isinstance(self.function, Callable):
            raise ValueError(f'Function must be callable. Got: {type(self.function)}')


    def y(self, x: float) -> float:
        return self.function(x)

    def _validate_initial_state(self) -> None:
        if not self.function:
            raise ValueError('Function not provided')
        elif not isinstance(self.function, sympy.Function):
            raise ValueError(f'Function must be a SymPy function. Got: {type(self.function)}')


    def initialize(self) -> None:
        slope_0 = self.y(self.x0)

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
        slope = self.y(xn)
        dx = self.dx
        return dict(
            x=xn + slope * dx,
            slope=slope,
            dx=dx
        )
