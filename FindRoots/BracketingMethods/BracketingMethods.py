from abc import ABC
from dataclasses import dataclass, field

import numpy as np

from FindRoots.RootFinder import RootFinder


@dataclass
class BracketingMethods(RootFinder, ABC):
    x_upper: float = field(default=None)
    x_lower: float = field(default=None)
    tolerance: float = field(default=1e-6)

    def _validate_initial_state(self) -> None:
        super()._validate_initial_state()
        if self.x_upper is None:
            raise ValueError('Initial state must include x_upper')
        if self.x_lower is None:
            raise ValueError('Initial state must include x_lower')

    @property
    def x_root(self) -> float:
        return self.roots[0]

    @x_root.setter
    def x_root(self, value: float) -> None:
        if len(self.roots) == 0:
            self.roots.append(value)
            return
        self.roots[0] = value

    def _is_x_root_converged(self) -> bool:
        return np.isclose(self.history.loc[self.last_iteration, 'x_root'], self.x_root, atol=self.tolerance)


