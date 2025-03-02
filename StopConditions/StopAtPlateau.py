from dataclasses import field, dataclass
from typing import Generator, Tuple

from Core.Numerical import StopCondition


@dataclass
class StopAtPlateau(StopCondition):
    tracking: str = field(default=None)
    patience: int = field(default=10)
    tolerance: float = field(default=1e-6)
    patience_counter: int = field(init=False, default=0)

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

            # Check if value has plateaued within tolerance
            if abs(current - previous) <= self.tolerance:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    self.stop_reason = (f"Variable '{self.tracking}' plateaued for"
                                        f" {self.patience} iterations within tolerance {self.tolerance}")
                    yield True, self.stop_reason
                    break
                yield False, (f"Potential plateau detected - {self.patience_counter}/{self.patience} "
                              f"iterations within tolerance {self.tolerance}")
            else:
                # Reset counter if significant change observed
                self.patience_counter = 0
                yield False, (f"Change detected in '{self.tracking}': {previous} → {current}, "
                              f"difference: {abs(current - previous)}")