from abc import ABC
from dataclasses import dataclass, field

from FindRoots.FindOneRoot import FindOneRoot


@dataclass
class BracketingMethods(FindOneRoot, ABC):
    x_upper: float = field(default=None)
    x_lower: float = field(default=None)

    def _validate_initial_state(self) -> None:
        super()._validate_initial_state()
        if self.x_upper is None:
            raise ValueError('Initial state must include x_upper')
        if self.x_lower is None:
            raise ValueError('Initial state must include x_lower')



