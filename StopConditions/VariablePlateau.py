from typing import Generator

from sympy import field

from Numerical import StopCondition


class VariablePlateau(StopCondition):
    var_name: str
    patience: int = field(ge=1, default=10)
    tolerance: float = field(gt=0, default=1e-6)
    patience_counter: int = field(init=False, default=0)

    def __post_init__(self):
        super().__post_init__()
        if self.var_name not in self.history:
            raise ValueError(f"Variable '{self.var_name}' not found in history."
                             f"Available variables: {', '.join(self.history.keys())}")
        self.stop_reason = f'Variable plateau of {self.var_name} after {self.patience} iterations'

    @property
    def last_iteration(self) -> int:
        return max(self.history[self.var_name].keys())

    def get_last_value(self) -> float:
        return self.history[self.var_name][self.last_iteration]


    def stop_condition_generator(self) -> Generator[bool, None, None]:
        while True:
            if self.patience_counter >= self.patience:
                break
            var_value = self.get_last_value()
            if abs(var_value) < self.tolerance:
                self.patience_counter += 1
            else:
                self.patience_counter = 0
            yield
        raise StopIteration