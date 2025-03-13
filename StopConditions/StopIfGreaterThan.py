from dataclasses import dataclass, field
from typing import Generator, Tuple

from StopConditions.StopConditionBase import StopCondition
from utils.ValidationTools import is_nan
from utils.log_config import get_logger


@dataclass
class StopIfGreaterThan(StopCondition):
    tracking: str = field(default=None, metadata=dict(help="Variable to track"))
    threshold: float = field(default=None, metadata=dict(help="Threshold value"))
    include_equal: bool = field(default=False, metadata=dict(help="Also stop if value equals threshold"))
    patience: int = field(default=1, metadata=dict(help="Number of consecutive iterations condition must be met"))
    patience_counter: int = field(init=False, default=0)

    def __post_init__(self):
        """Validate initialization parameters."""
        super().__post_init__()
        if self.tracking is None:
            raise ValueError("Must specify tracking variable")
        if self.threshold is None:
            raise ValueError("Must specify threshold value")
        if self.patience < 0:
            raise ValueError(f"Patience must be positive, got {self.patience}")
        self.patience_counter = 0  # Initialize counter
        self.logger = get_logger(self.__class__.__name__)

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}(tracking='{self.tracking}', threshold={self.threshold}, include_equal={self.include_equal}, patience={self.patience}, patience_counter={self.patience_counter})"

    def __str__(self):
        """Return a concise, human-readable description of the StopIfGreaterThan condition."""
        class_name = self.__class__.__name__
        comparison = ">=" if self.include_equal else ">"
        return f"{class_name}: Stop when '{self.tracking}' {comparison} {self.threshold:.6g} for {self.patience} iterations"

    def stop_condition_generator(self) -> Generator[Tuple[bool, str], None, None]:
        while True:
            current = self.history[self.tracking]

            # Handle NaN values
            if is_nan(current):
                self.logger.warning(f"Variable {self.tracking} is NaN")
                yield True, f"Variable {self.tracking} is NaN"
                continue

            # Calculate difference from threshold
            diff = current - self.threshold

            # Update history with detailed metrics
            self.update_stop_history(dict(
                current_value=current,
                threshold=self.threshold,
                diff=diff,
                include_equal=self.include_equal
            ))

            # Check condition based on include_equal flag
            condition_met = current > self.threshold or (self.include_equal and current == self.threshold)
            comparison_symbol = ">=" if self.include_equal else ">"

            if condition_met:
                self.patience_counter += 1

                if self.patience_counter >= self.patience:
                    self.stop_reason = (
                        f"Variable {self.tracking}:{float(current):.6g} {comparison_symbol} "
                        f"threshold {self.threshold:.6g} "
                        f"for {self.patience} iterations"
                    )
                    self.logger.debug(self.stop_reason)
                    yield True, self.stop_reason
                else:
                    continue_reason = (
                        f"Variable {self.tracking}:{float(current):.6g} {comparison_symbol} "
                        f"threshold {self.threshold:.6g} "
                        f"({self.patience_counter}/{self.patience} iterations)"
                    )
                    self.logger.debug(continue_reason)
                    yield False, continue_reason
            else:
                # Reset counter if value doesn't meet condition
                self.patience_counter = 0

                inverse_symbol = "<" if self.include_equal else "<="
                continue_reason = (
                    f"Variable {self.tracking}:{float(current):.6g} {inverse_symbol} "
                    f"{self.threshold:.6g} (diff: {float(diff):.6g})"
                )
                self.logger.debug(continue_reason)
                yield False, continue_reason