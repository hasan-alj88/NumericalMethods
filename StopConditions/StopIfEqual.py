from dataclasses import dataclass, field
from typing import Generator, Tuple

from StopConditions.StopConditionBase import StopCondition
from utils.ValidationTools import is_nan
from utils.log_config import get_logger


@dataclass
class StopIfEqual(StopCondition):
    tracking: str = field(default=None)
    value: float = field(default=0.0)
    tolerance: float = field(default=1e-6)
    patience: int = field(default=3)
    patience_counter: int = field(init=False, default=0)

    def __post_init__(self):
        """Validate initialization parameters."""
        if self.tracking is None:
            raise ValueError("Must specify a variable name to track")
        if self.tolerance <= 0:
            raise ValueError(f"Absolute tolerance must be positive, got {self.tolerance}")
        if self.patience <= 2:
            raise ValueError(f"Patience must be >1, got {self.patience}")
        self.patience_counter = 0  # Initialize counter
        self.logger = get_logger(self.__class__.__name__)

    def stop_condition_generator(self) -> Generator[Tuple[bool, str], None, None]:
        while True:
            current = self.history[self.tracking]

            # Handle NaN values
            if is_nan(current):
                self.logger.warning(f"Variable {self.tracking} is NaN")
                yield True, f"Variable {self.tracking} is NaN"
                continue

            # Check if we've reached target
            diff = abs(current - self.value)
            if diff <= self.tolerance:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    self.stop_reason = (
                        f"Variable {self.tracking}:{current:0.3e} reached "
                        f"target {self.value:0.3e} within "
                        f"tolerance {self.tolerance:0.3e} for {self.patience} iterations"
                    )
                    self.logger.debug(self.stop_reason)
                    yield True, self.stop_reason
                else:
                    continue_reason = (
                        f"Variable {self.tracking}:{current:0.3e} matches "
                        f"target {self.value:0.3e} within tolerance {self.tolerance} "
                        f"({self.patience_counter}/{self.patience} iterations)"
                    )
                    self.logger.debug(continue_reason)
                    yield False, continue_reason
            else:
                # Reset counter if value doesn't match
                self.patience_counter = 0
                continue_reason = (
                    f"Variable {self.tracking}:{current:0.3e} != "
                    f"{self.value:0.3e} (diff: {diff:0.3e})"
                )
                self.logger.debug(continue_reason)
                yield False, continue_reason

class StopIfZero(StopIfEqual):
    def __init__(self, *args, **kwargs):
        self.value: float = 0.0
        super().__init__(*args, **kwargs)