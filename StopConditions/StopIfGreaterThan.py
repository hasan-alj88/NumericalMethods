from dataclasses import field, dataclass
from typing import Generator, Tuple

from StopConditions.StopConditionBase import StopCondition


@dataclass
class StopIfGreaterThan(StopCondition):
    tracking: str = field(default=None, metadata=dict(help="Variable to track"))
    threshold: float = field(default=None, metadata=dict(help="Threshold value"))

    def __post_init__(self):
        if self.tracking is None:
            raise ValueError("Must specify tracking variable")
        if self.threshold is None:
            raise ValueError("Must specify threshold value")

    def stop_condition_generator(self) -> Generator[Tuple[bool, str], None, None]:
        while True:
            current = self.history[self.tracking]
            if current > self.threshold:
                yield True, f"Variable {self.tracking} is greater than {self.threshold}"
                break
            else:
                yield False, f"Variable {self.tracking} is less than {self.threshold}"