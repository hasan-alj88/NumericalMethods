from dataclasses import dataclass, field
from typing import Generator, Tuple, List

from StopConditions.StopConditionBase import StopCondition
from utils.ValidationTools import is_nan


@dataclass
class StopIfNaN(StopCondition):
    track_variables: List[str] = field(default_factory=list)

    def __post_init__(self):
        if len(self.track_variables) == 0:
            raise ValueError("Must specify at least one variable to track")

    def __repr__(self):
        return f"StopIfNaN(track_variables={self.track_variables})"

    def __str__(self):
        return f"StopIfNaN: Stop if any of {self.track_variables} is NaN"

    def stop_condition_generator(self) -> Generator[Tuple[bool, str], None, None]:
        while True:
            for var in self.track_variables:
                if is_nan(self.history[var]):
                    yield True, f"Variable {var} is NaN"
                    break
            else:
                yield False, "No NaN values found"

