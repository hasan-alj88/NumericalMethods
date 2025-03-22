from decimal import Decimal
from typing import List, Tuple, Optional, Dict, Any
from pydantic.v1 import BaseModel, Field, validator

from utils.ErrorCalculations import tolerance_condition


class StopAtPlateau(BaseModel):
    tracking: str
    patience: int = 10
    absolute_tolerance: Optional[Decimal] = None
    relative_tolerance: Optional[Decimal] = None
    patience_counter: int = Field(default=0, init=False)
    stop_history: List[Dict[str, Any]] = Field(default_factory=list, init=False)

    @validator('tracking')
    def validate_tracking(cls, v):
        if not v:
            raise ValueError("Must specify a variable name to track")
        return v

    @validator('patience')
    def validate_patience(cls, v):
        if v < 0:
            raise ValueError(f"Patience must be positive, got {v}")
        return v

    @validator('absolute_tolerance', 'relative_tolerance')
    def validate_tolerances(cls, v, values):
        # This validator will be called twice - once for each field
        # We need to check if both are None only after both have been processed
        if 'absolute_tolerance' in values and 'relative_tolerance' in values:
            if values['absolute_tolerance'] is None and values['relative_tolerance'] is None:
                raise ValueError("Must specify either absolute_tolerance or relative_tolerance")
        return v

    def update_stop_history(self, data: Dict[str, Any]) -> None:
        """Update the history of stop condition evaluations"""
        self.stop_history.append(data)

    def is_stop(self, history: List[Any]) -> Tuple[bool, str]:
        """
        Check if the tracked variable has plateaued.

        Args:
            history: List of state objects with the tracked variable

        Returns:
            Tuple of (bool, str): Whether to stop and log message
        """
        # Need at least 2 iterations to check for plateaus
        if len(history) < 2:
            return False, "Not enough iterations to determine plateau"

        # Get current and previous values
        try:
            current = getattr(history[-1], self.tracking)
            previous = getattr(history[-2], self.tracking)
        except AttributeError:
            raise ValueError(f"Could not access '{self.tracking}' in history objects."
                             f"Available attributes: {history[-1].__dict__.keys()}")

        # Stop condition logic: if either tolerance is met (or both)
        stop_condition_state = tolerance_condition(
            first_value=Decimal(previous),
            second_value=Decimal(current),
            absolute_tolerance=self.absolute_tolerance,
            relative_tolerance=self.relative_tolerance
        )
        stop_condition = stop_condition_state['stop_condition']
        abs_diff = stop_condition_state['abs_diff']
        rel_diff = stop_condition_state['rel_diff']
        self.update_stop_history(stop_condition_state)

        # Check if plateau has been sustained
        if stop_condition:
            self.patience_counter += 1

            # Create a tolerance description that includes the active tolerance(s)
            tolerance_desc = []
            if self.absolute_tolerance is not None:
                tolerance_desc.append(f"abs diff: {abs_diff:.6g} ≤ {self.absolute_tolerance:.6g}")
            if self.relative_tolerance is not None:
                tolerance_desc.append(f"rel diff: {rel_diff:.6g} ≤ {self.relative_tolerance:.6g}")
            tolerance_str = " or ".join(tolerance_desc)

            if self.patience_counter >= self.patience:
                stop_reason = (f"Variable '{self.tracking}' plateaued for"
                               f" {self.patience} iterations ({tolerance_str})")
                return True, stop_reason
            return False, (f"Potential plateau detected - {self.patience_counter}/{self.patience} "
                           f"iterations ({tolerance_str})")
        else:
            # Reset counter if significant change observed
            self.patience_counter = 0

            # Create message showing why condition wasn't met
            if self.absolute_tolerance is not None and self.relative_tolerance is not None:
                reason = (f"Change detected in '{self.tracking}': {previous:.6g} → {current:.6g}, "
                          f"abs diff: {abs_diff:.6g} > {self.absolute_tolerance:.6g} and "
                          f"rel diff: {rel_diff:.6g} > {self.relative_tolerance:.6g}")
            elif self.absolute_tolerance is not None:
                reason = (f"Change detected in '{self.tracking}': {previous:.6g} → {current:.6g}, "
                          f"abs diff: {abs_diff:.6g} > {self.absolute_tolerance:.6g}")
            else:
                reason = (f"Change detected in '{self.tracking}': {previous:.6g} → {current:.6g}, "
                          f"rel diff: {rel_diff:.6g} > {self.relative_tolerance:.6g}")

            return False, reason
