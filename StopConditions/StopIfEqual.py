from dataclasses import dataclass, field
from typing import Generator, Tuple, Optional

from StopConditions.StopConditionBase import StopCondition
from utils.ValidationTools import is_nan
from utils.log_config import get_logger


@dataclass
class StopIfEqual(StopCondition):
    tracking: str = field(default=None)
    value: float | str = field(default=0.0)
    absolute_tolerance: Optional[float] = field(default=None)
    relative_tolerance: Optional[float] = field(default=None)
    patience: int = field(default=3)
    patience_counter: int = field(init=False, default=0)

    def __post_init__(self):
        """Validate initialization parameters."""
        if self.tracking is None:
            raise ValueError("Must specify a variable name to track")

        if self.absolute_tolerance is None and self.relative_tolerance is None:
            raise ValueError("Must specify either absolute_tolerance or relative_tolerance")

        if self.absolute_tolerance is not None and self.absolute_tolerance <= 0:
            raise ValueError(f"Absolute tolerance must be positive, got {self.absolute_tolerance}")

        if self.relative_tolerance is not None and self.relative_tolerance <= 0:
            raise ValueError(f"Relative tolerance must be positive, got {self.relative_tolerance}")

        if self.patience < 0:
            raise ValueError(f"Patience must be positive, got {self.patience}")
        self.patience_counter = 0  # Initialize counter
        self.logger = get_logger(self.__class__.__name__)

    def __repr__(self):
        class_name = self.__class__.__name__
        value_repr = f"'{self.value}'" if isinstance(self.value, str) else f"{self.value}"

        # Format tolerances
        abs_tol = f"{self.absolute_tolerance}" if self.absolute_tolerance is not None else "None"
        rel_tol = f"{self.relative_tolerance}" if self.relative_tolerance is not None else "None"

        return (f"{class_name}(tracking='{self.tracking}', value={value_repr}, "
                f"absolute_tolerance={abs_tol}, relative_tolerance={rel_tol}, "
                f"patience={self.patience}, patience_counter={self.patience_counter})")

    def __str__(self):
        """Return a concise, human-readable description of the StopIfEqual condition.

        This provides a summary of what the condition is checking for.
        """
        class_name = self.__class__.__name__
        value_str = f"'{self.value}'" if isinstance(self.value, str) else f"{self.value:.6g}"

        # Build tolerance description
        tol_parts = []
        if self.absolute_tolerance is not None:
            tol_parts.append(f"abs_tol={self.absolute_tolerance:.6g}")
        if self.relative_tolerance is not None:
            tol_parts.append(f"rel_tol={self.relative_tolerance:.6g}")
        tol_str = " or ".join(tol_parts)

        return (f"{class_name}: Stop when '{self.tracking}' equals {value_str} "
                f"({tol_str}) for {self.patience} iterations")

    def get_value(self):
        if isinstance(self.value, str):
            return self.history[self.value]
        else:
            return self.value

    def stop_condition_generator(self) -> Generator[Tuple[bool, str], None, None]:
        while True:
            current = self.history[self.tracking]
            value = self.get_value()

            # Handle NaN values
            if is_nan(current):
                self.logger.warning(f"Variable {self.tracking} is NaN")
                yield True, f"Variable {self.tracking} is NaN"
                continue
            if is_nan(value):
                self.logger.warning(f"Value {value} is NaN")
                yield True, f"Value {value} is NaN"
                continue

            # Calculate absolute and relative differences
            abs_diff = abs(current - value)

            # Only calculate relative difference if value is not zero
            rel_diff = abs_diff / abs(value) if value != 0 else float('inf')

            # Determine if tolerances are met
            abs_tol_met = self.absolute_tolerance is not None and abs_diff <= self.absolute_tolerance
            rel_tol_met = self.relative_tolerance is not None and rel_diff <= self.relative_tolerance

            # Update history with detailed metrics
            self.update_stop_history(dict(
                abs_diff=abs_diff,
                rel_diff=rel_diff,
                abs_tol_met=abs_tol_met,
                rel_tol_met=rel_tol_met
            ))

            # Check if either tolerance condition is met
            if abs_tol_met or rel_tol_met:
                self.patience_counter += 1

                # Create tolerance description
                tolerance_desc = []
                if abs_tol_met:
                    tolerance_desc.append(f"abs diff: {float(abs_diff):.6g} ≤ {self.absolute_tolerance:.6g}")
                if rel_tol_met:
                    tolerance_desc.append(f"rel diff: {float(rel_diff):.6g} ≤ {self.relative_tolerance:.6g}")
                tolerance_str = " or ".join(tolerance_desc)

                if self.patience_counter >= self.patience:
                    self.stop_reason = (
                        f"Variable {self.tracking}:{float(current):.6g} reached "
                        f"target {value:.6g} ({tolerance_str}) "
                        f"for {self.patience} iterations"
                    )
                    self.logger.debug(self.stop_reason)
                    yield True, self.stop_reason
                else:
                    continue_reason = (
                        f"Variable {self.tracking}:{float(current):.6g} matches "
                        f"target {value:.6g} ({tolerance_str}) "
                        f"({self.patience_counter}/{self.patience} iterations)"
                    )
                    self.logger.debug(continue_reason)
                    yield False, continue_reason
            else:
                # Reset counter if value doesn't match within tolerances
                self.patience_counter = 0

                # Create message showing why condition wasn't met
                condition_desc = []
                if self.absolute_tolerance is not None:
                    condition_desc.append(f"abs diff: {float(abs_diff):.6g} > {self.absolute_tolerance:.6g}")
                if self.relative_tolerance is not None:
                    condition_desc.append(f"rel diff: {float(rel_diff):.6g} > {self.relative_tolerance:.6g}")
                condition_str = " and ".join(condition_desc)

                continue_reason = (
                    f"Variable {self.tracking}:{float(current):.6g} != "
                    f"{value:.6g} ({condition_str})"
                )
                self.logger.debug(continue_reason)
                yield False, continue_reason


class StopIfZero(StopIfEqual):
    def __init__(self, *args, **kwargs):
        self.value: float = 0.0
        super().__init__(*args, **kwargs)