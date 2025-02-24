from abc import ABC
from dataclasses import dataclass, field

import numpy as np

from FindRoots.RootFinder import RootFinder


@dataclass
class FindOneRoot(RootFinder, ABC):
    tolerance: float = field(default=1e-6)

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
        return np.isclose(0.0, self.x_root, atol=self.tolerance)

