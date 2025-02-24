from dataclasses import dataclass, field
from typing import Generator, Tuple

import numpy as np

from Numerical import StopCondition


@dataclass
class StopIfEqual(StopCondition):
    tracking: str = field(default=None)
    value: float = field(default=0.0)
    tolerance: float = field(default=1e-6)

    def __post_init__(self):
        """Validate initialization parameters."""
        if self.tracking is None:
            raise ValueError("Must specify a variable name to track")
        if self.tolerance <= 0:
            raise ValueError(f"Absolute tolerance must be positive, got {self.tolerance}")

    def stop_condition_generator(self) -> Generator[Tuple[bool, str], None, None]:
        while True:

            current = self.history.loc[self.last_iteration, self.tracking]

            # Handle NaN values
            if np.isnan(current):
                yield True, f"Variable {self.tracking} is NaN"
                continue

            # Check if we've reached target
            if np.isclose(current, self.value, atol=self.tolerance):
                yield True, (
                    f"Variable {self.tracking}:{current:0.3e} reached "
                    f"target {self.value:0.3e} witin"
                    f"tolerance {self.tolerance:0.3e}"
                )

            yield False, (
                f"Variable {self.tracking}:{current:0.3e} != "
                f"{self.value:0.3e} (diff: {abs(current - self.value):0.3e})"
            )

class StopIfZero(StopIfEqual):
    def __init__(self, *args, **kwargs):
        self.value: float = 0.0
        super().__init__(*args, **kwargs)
