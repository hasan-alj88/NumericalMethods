from dataclasses import field, dataclass
from typing import Generator, Tuple, Optional

from Core.Numerical import StopCondition


@dataclass
class StopAtPlateau(StopCondition):
    tracking: str = field(default=None)
    patience: int = field(default=10)
    absolute_tolerance: Optional[float] = field(default=None)
    relative_tolerance: Optional[float] = field(default=None)
    patience_counter: int = field(init=False, default=0)

    def __post_init__(self):
        """Validate initialization parameters."""
        if self.tracking is None:
            raise ValueError("Must specify a variable name to track")
        if self.patience < 0:
            raise ValueError(f"Patience must be positive, got {self.patience}")

        if self.absolute_tolerance is None and self.relative_tolerance is None:
            raise ValueError("Must specify either tolerance or relative tolerance")

    def __repr__(self):
        """Return a detailed string representation of the StopAtPlateau object.

        This provides all parameters and their values for debugging purposes.
        """
        class_name = self.__class__.__name__

        # Format tolerances
        abs_tol = f"{self.absolute_tolerance}" if self.absolute_tolerance is not None else "None"
        rel_tol = f"{self.relative_tolerance}" if self.relative_tolerance is not None else "None"

        return (f"{class_name}(tracking='{self.tracking}', "
                f"absolute_tolerance={abs_tol}, relative_tolerance={rel_tol}, "
                f"patience={self.patience}, patience_counter={self.patience_counter})")

    def __str__(self):
        """Return a concise, human-readable description of the StopAtPlateau condition.

        This provides a summary of what the condition is checking for.
        """
        class_name = self.__class__.__name__

        # Build tolerance description
        tol_parts = []
        if self.absolute_tolerance is not None:
            tol_parts.append(f"abs_tol={self.absolute_tolerance:.6g}")
        if self.relative_tolerance is not None:
            tol_parts.append(f"rel_tol={self.relative_tolerance:.6g}")

        # Check if we need "and" or "or" based on the implementation logic
        # In stop_condition_generator, the condition uses "or" when both tolerances are specified
        tol_str = " or ".join(tol_parts)

        return (f"{class_name}: Stop when '{self.tracking}' plateaus "
                f"({tol_str}) for {self.patience} iterations")

    def stop_condition_generator(self) -> Generator[Tuple[bool, str], None, None]:
        """Generate stop condition based on plateau detection in tracked variable"""
        self.patience_counter = 0

        while True:
            # Need at least 2 iterations to check for plateaus
            if len(self.history) < 2:
                yield False, f"Not enough iterations to determine plateau"
                continue

            # Get current and previous values
            current = self.history[self.tracking]
            previous = self.history(-2, self.tracking)

            abs_diff = abs(current - previous)
            rel_diff = abs_diff / abs(previous) if previous != 0 else float('inf')

            stop_condition = ((self.absolute_tolerance is not None and self.relative_tolerance is not None) and
                              (abs_diff <= self.absolute_tolerance or rel_diff <= self.relative_tolerance))
            self.update_stop_history(dict(
                abs_diff=abs_diff,
                rel_diff=rel_diff,
                abs_tol_met=(self.absolute_tolerance is not None and abs_diff <= self.absolute_tolerance),
                rel_tol_met=(self.relative_tolerance is not None and rel_diff <= self.relative_tolerance),
                stop_condition=stop_condition
            ))

            # Check if value has plateaued within tolerance
            if stop_condition:
                self.patience_counter += 1

                # Create a tolerance description that includes the active tolerance(s)
                tolerance_desc = []
                if self.absolute_tolerance is not None:
                    tolerance_desc.append(f"abs diff: {abs_diff:.6g} ≤ {self.absolute_tolerance:.6g}")
                if self.relative_tolerance is not None:
                    tolerance_desc.append(f"rel diff: {rel_diff:.6g} ≤ {self.relative_tolerance:.6g}")
                tolerance_str = " and ".join(tolerance_desc)

                if self.patience_counter >= self.patience:
                    self.stop_reason = (f"Variable '{self.tracking}' plateaued for"
                                        f" {self.patience} iterations ({tolerance_str})")
                    yield True, self.stop_reason
                    break
                yield False, (f"Potential plateau detected - {self.patience_counter}/{self.patience} "
                              f"iterations ({tolerance_str})")
            else:
                # Reset counter if significant change observed
                self.patience_counter = 0

                # Create message showing why condition wasn't met
                if self.absolute_tolerance is not None and self.relative_tolerance is not None:
                    reason = (f"Change detected in '{self.tracking}': {previous:.6g} → {current:.6g}, "
                              f"abs diff: {abs_diff:.6g} > {self.absolute_tolerance:.6g} or "
                              f"rel diff: {rel_diff:.6g} > {self.relative_tolerance:.6g}")
                elif self.absolute_tolerance is not None:
                    reason = (f"Change detected in '{self.tracking}': {previous:.6g} → {current:.6g}, "
                              f"abs diff: {abs_diff:.6g} > {self.absolute_tolerance:.6g}")
                else:
                    reason = (f"Change detected in '{self.tracking}': {previous:.6g} → {current:.6g}, "
                              f"rel diff: {rel_diff:.6g} > {self.relative_tolerance:.6g}")

                yield False, reason