from abc import ABC
from dataclasses import dataclass, field
from typing import Tuple

from FindRoots.RootFinder import RootFinder


@dataclass
class BracketingMethods(RootFinder, ABC):
    a: float = field(default=None)
    b: float = field(default=None)

    def __post_init__(self):
        fa = self.function(self.a)
        fb = self.function(self.b)
        if fa * fb > 0:
            raise ValueError(f"f(a) and f(b) must have opposite signs. Got {fa} and {fb}")


    @classmethod
    def initial_range(cls, initial_range: Tuple[float,float], *args, **kwargs) -> 'BracketingMethods':
        return cls(b=max(initial_range), a=min(initial_range), *args, **kwargs)

    def _validate_initial_state(self) -> None:
        super()._validate_initial_state()
        if self.a is None:
            raise ValueError('Initial state must include a (lower bound)')
        if self.b is None:
            raise ValueError('Initial state must include b (upper bound)')

        self.b, self.a = max(self.b, self.a), min(self.b, self.a)
