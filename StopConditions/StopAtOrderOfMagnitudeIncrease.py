import math
from typing import Tuple, Optional, List, Any

from pydantic.v1 import BaseModel, Field, validator

from utils.ValidationTools import is_nan


class StopAtOrderOfMagnitudeIncrease(BaseModel):
    tracking: str
    patience: int = 1
    orders_increase: int = 1  # Number of orders of magnitude to trigger (default: 1 order = 10x)
    adapt_to_decrease: bool = True
    patience_counter: int = Field(init=False, default=0)
    baseline_order: Optional[int] = Field(init=False, default=None)
    baseline_value: Optional[float] = Field(init=False, default=None)
    triggered: bool = Field(init=False, default=False)
    stop_history: list = Field(default_factory=list, init=False)

    @validator('tracking')
    def validate_tracking(cls, v): # noqa
        if v is None:
            raise ValueError("Must specify a variable name to track")
        return v

    @validator('patience')
    def validate_patience(cls, v): # noqa
        if v < 1:
            raise ValueError(f"Patience must be at least 1, got {v}")
        return v

    @validator('orders_increase')
    def validate_orders_increase(cls, v): # noqa
        if v < 1:
            raise ValueError(f"Orders of magnitude increase must be at least 1, got {v}")
        return v

    def __repr__(self):
        """Return a detailed string representation of the object."""
        class_name = self.__class__.__name__
        return (f"{class_name}(tracking='{self.tracking}', "
                f"orders_increase={self.orders_increase}, "
                f"patience={self.patience}, patience_counter={self.patience_counter}, "
                f"baseline_order={self.baseline_order}, baseline_value={self.baseline_value}, "
                f"triggered={self.triggered}, adapt_to_decrease={self.adapt_to_decrease})")

    def __str__(self):
        """Return a concise, human-readable description of the stop condition."""
        class_name = self.__class__.__name__
        magnitude_str = f"10^{self.orders_increase}" if self.orders_increase == 1 else f"10^{self.orders_increase}"
        adapt_str = "with adaptive baseline" if self.adapt_to_decrease else ""
        return (f"{class_name}: Stop when '{self.tracking}' increases by {magnitude_str} "
                f"and doesn't return for {self.patience} iterations {adapt_str}")


    def _get_order_of_magnitude(self, value):
        """Calculate the order of magnitude (floor of log10) of a value."""
        if is_nan(value) or value <= 0:
            return 10**(self.baseline_order+2)
        return math.floor(math.log10(value))

    def is_stop(self, history: List[Any]) -> Tuple[bool, str]:
        """Generate stop condition based on integer order of magnitude increase detection."""

        if len(history) < 1:
            return False, f"Not enough iterations to determine baseline value"

        try:
            current = getattr(history[-1], self.tracking)
        except AttributeError:
            raise ValueError(f"Could not access '{self.tracking}' in history objects."
                             f"Available attributes: {history[-1].__dict__.keys()}")

        # Handle special case for zero or negative values
        if current <= 0:
            return False, f"Cannot determine order of magnitude for non-positive value: {current}"

        current_order = self._get_order_of_magnitude(current)

        # Initialize baseline on first iteration
        if self.baseline_value is None:
            self.baseline_value = float(current)
            self.baseline_order = current_order
            return False, f"Baseline established: {self.baseline_value} (order: 10^{self.baseline_order})"

        # Update baseline if value decreases significantly and we're not in triggered state
        if self.adapt_to_decrease and not self.triggered and current_order < self.baseline_order:
            old_baseline = self.baseline_value
            old_order = self.baseline_order
            self.baseline_value = current
            self.baseline_order = current_order
            return False, (f"Baseline decreased from {old_baseline:.6g} (10^{old_order}) to "
                          f"{self.baseline_value} (10^{self.baseline_order})")

        # Check if order of magnitude trigger condition is met
        target_order = self.baseline_order + self.orders_increase
        if not self.triggered and current_order >= target_order:
            self.triggered = True
            return False, (f"Order of magnitude increase detected: current={current:.6g} (10^{current_order}) ≥ "
                          f"threshold=10^{target_order} (baseline: {self.baseline_value:.6g}, 10^{self.baseline_order})")

        # If triggered, check if value returned below threshold
        if self.triggered:
            target_order = self.baseline_order + self.orders_increase

            # If value returns below threshold, reset
            if current_order < target_order:
                self.triggered = False
                self.patience_counter = 0
                # Update baseline to current value
                self.baseline_value = current
                self.baseline_order = current_order
                return False, (f"Value returned below threshold: {current:.6g} (10^{current_order}) < "
                              f"10^{target_order}, resetting baseline")
            else:
                # Value still above threshold, increment counter
                self.patience_counter += 1

                # Create status update
                self.stop_history.append(dict(
                    current=current,
                    current_order=current_order,
                    baseline=self.baseline_value,
                    baseline_order=self.baseline_order,
                    threshold_order=target_order,
                    patience_counter=self.patience_counter,
                    triggered=self.triggered
                ))

                if self.patience_counter >= self.patience:
                    magnitude_diff = current_order - self.baseline_order
                    return True, (
                        f"Variable '{self.tracking}' increased by {magnitude_diff} order(s) of magnitude "
                        f"from {self.baseline_value:.6g} (10^{self.baseline_order}) to "
                        f"{current:.6g} (10^{current_order}) and remained elevated for {self.patience} iterations")

                return False, (f"Value remains elevated: {current:.6g} (10^{current_order}) ≥ "
                              f"10^{target_order}, {self.patience_counter}/{self.patience} iterations")
        else:
            # Not triggered, but no significant change - continue monitoring
            return False, f"Monitoring: current={current:.6g} (10^{current_order}), baseline={self.baseline_value:.6g} (10^{self.baseline_order})"