from dataclasses import Field
from decimal import Decimal
from typing import Tuple, Optional, Dict, List, Any

from pydantic.v1 import BaseModel, validator

from utils.ErrorCalculations import tolerance_condition


class StopIfEqual(BaseModel):
    tracking: str
    value: Decimal = Decimal("0")
    absolute_tolerance: Optional[Decimal]
    relative_tolerance: Optional[Decimal]
    patience: int = 1
    patience_counter: int = Field(init=False, default=0)
    stop_history: list = Field(default_factory=list, init=False)

    @validator('tracking')
    def validate_tracking(cls, v): # noqa
        if v is None:
            raise ValueError("Must specify a variable name to track")
        return v

    @validator('absolute_tolerance', 'relative_tolerance')
    def validate_tolerances(cls, v, values): # noqa
        # This validator will be called twice - once for each field
        # We need to check if both are None only after both have been processed
        abs_tol = values.get('absolute_tolerance')
        rel_tol = values.get('relative_tolerance')
        if abs_tol is None and rel_tol is None:
            raise ValueError("Must specify either absolute_tolerance or relative_tolerance")
        if abs_tol and abs_tol <= 0:
            raise ValueError(f"Absolute tolerance must be positive, got {abs_tol}")
        if rel_tol and rel_tol <= 0:
            raise ValueError(f"Relative tolerance must be positive, got {rel_tol}")
        return v

    @validator('value')
    def validate_value(cls, v): # noqa
        if v is None:
            raise ValueError("Must specify a value to compare against")
        return v

    @validator('patience')
    def validate_patience(cls, v): # noqa
        if v < 0:
            raise ValueError(f"Patience must be positive, got {v}")


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

    def update_stop_history(self, data: Dict[str, Any]) -> None:
        """Update the history of stop condition evaluations"""
        self.stop_history.append(data)

    def is_stop(self, history: List[Any]) -> Tuple[bool, str]:

        if len(history) < 1:
            return False, "Not enough iterations to determine stop condition"

        try:
            current = getattr(history[-1], self.tracking)
        except AttributeError:
            raise ValueError(f"Could not access '{self.tracking}' in history objects."
                             f"Available attributes: {history[-1].__dict__.keys()}")

        stop_condition_info = tolerance_condition(
            first_value=Decimal(self.value),
            second_value=Decimal(current),
            absolute_tolerance=self.absolute_tolerance,
            relative_tolerance=self.relative_tolerance
        )
        self.update_stop_history(stop_condition_info)
        abs_tol_met = stop_condition_info['abs_tol_met']
        rel_tol_met = stop_condition_info['rel_tol_met']
        abs_diff = stop_condition_info['abs_diff']
        rel_diff = stop_condition_info['rel_diff']

        # Check if either tolerance condition is met
        if abs_tol_met or rel_tol_met:
            self.patience_counter += 1

            # Create tolerance description
            tolerance_desc = []
            if abs_tol_met:
                tolerance_desc.append(f"abs diff: {abs_diff:.6g} ≤ {self.absolute_tolerance:.6g}")
            if rel_tol_met:
                tolerance_desc.append(f"rel diff: {rel_diff:.6g} ≤ {self.relative_tolerance:.6g}")
            tolerance_str = " or ".join(tolerance_desc)

            if self.patience_counter >= self.patience:
                return True, (
                    f"Variable {self.tracking}:{current:.6g} reached "
                    f"target {self.value:.6g} ({tolerance_str}) "
                    f"for {self.patience} iterations"
                )
            else:
                yield False, (
                    f"Variable {self.tracking}:{current:.6g} matches "
                    f"target {self.value:.6g} ({tolerance_str}) "
                    f"({self.patience_counter}/{self.patience} iterations)"
                )
        else:
            # Reset counter if value doesn't match within tolerances
            self.patience_counter = 0

            # Create message showing why condition wasn't met
            condition_desc = []
            if self.absolute_tolerance is not None:
                condition_desc.append(f"abs diff: {abs_diff:.6g} > {self.absolute_tolerance:.6g}")
            if self.relative_tolerance is not None:
                condition_desc.append(f"rel diff: {rel_diff:.6g} > {self.relative_tolerance:.6g}")
            condition_str = " and ".join(condition_desc)

            yield False, (
                f"Variable {self.tracking}:{current:.6g} != "
                f"{self.value:.6g} ({condition_str})"
            )
