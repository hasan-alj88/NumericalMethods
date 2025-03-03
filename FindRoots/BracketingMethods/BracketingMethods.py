from abc import ABC
from dataclasses import dataclass, field
from FindRoots.RootFinder import RootFinder


@dataclass
class BracketingMethods(RootFinder, ABC):
    t_upper: float = field(default=None)
    t_lower: float = field(default=None)

    def _validate_initial_state(self) -> None:
        super()._validate_initial_state()
        if self.t_upper is None:
            raise ValueError('Initial state must include t_upper')
        if self.t_lower is None:
            raise ValueError('Initial state must include t_lower')
