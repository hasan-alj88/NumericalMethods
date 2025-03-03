from abc import ABC
from dataclasses import dataclass, field
from typing import Tuple

from FindRoots.RootFinder import RootFinder


@dataclass
class BracketingMethods(RootFinder, ABC):
    t_upper0: float = field(default=None)
    t_lower0: float = field(default=None)

    @classmethod
    def initial_range(cls, initial_range: Tuple[float,float], *args, **kwargs) -> 'BracketingMethods':
        return cls(t_upper0=max(initial_range), t_lower0=min(initial_range), *args, **kwargs)

    def _validate_initial_state(self) -> None:
        super()._validate_initial_state()
        if self.t_upper0 is None:
            raise ValueError('Initial state must include t_upper0')
        if self.t_lower0 is None:
            raise ValueError('Initial state must include t_lower0')

        self.t_upper, self.t_lower = max(self.t_upper0, self.t_lower0), min(self.t_upper0, self.t_lower0)
