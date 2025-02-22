from dataclasses import dataclass
from typing import Generator

from Numerical import StopCondition


@dataclass
class VariableGreaterThan(StopCondition): # noqa
    tracked_variable: str
    threshold: float

    def __post_init__(self):
        if self.tracked_variable not in self.history.columns:
            raise ValueError(f"Variable '{self.tracked_variable}' not found in history."
                             f"Available variables: {', '.join(self.history.columns.tolist())}")
        self.stop_reason = f'Variable {self.tracked_variable} greater than {self.threshold}'

    def track_variable_value(self) -> float:
        return self.history.loc[self.last_iteration, self.tracked_variable]

    def stop_condition_generator(self) -> Generator[str, None, None]:
        while True:
            if self.track_variable_value() >= self.threshold:
                break
            yield f'Continue, Since {self.tracked_variable} < {self.track_variable_value()}'
