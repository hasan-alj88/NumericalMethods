from abc import ABC
from dataclasses import dataclass, field
from typing import Tuple

from FindRoots.RootFinder import RootFinder


@dataclass
class BracketingMethods(RootFinder, ABC):
    a: float = field(default=None)
    b: float = field(default=None)

    @classmethod
    def initial_range(cls, initial_range: Tuple[float,float], *args, **kwargs) -> 'BracketingMethods':
        return cls(b=max(initial_range), a=min(initial_range), *args, **kwargs)

    def _validate_initial_state(self) -> None:
        super()._validate_initial_state()
        if self.b is None:
            raise ValueError('Initial state must include t_upper0')
        if self.a is None:
            raise ValueError('Initial state must include t_lower0')

        self.t_upper, self.t_lower = max(self.b, self.a), min(self.b, self.a)
