import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Any

from Numerical import Numerical


@dataclass
class ExplicitFirstOrderMethod(Numerical, ABC):
    """
    Abstract base class for numerical methods for solving explicit first-order ODEs.
    dx/dt = f(t)
    - explicit: dx/dt can be explicitly calculated from the current state f(t).
    no terms with mix dependent and independent variables.
    - first-order: height derivative is 1.
    - Linear: Height derivative powers is 1.
    """
    derivative_function: Callable = field(default=None)
    t0: float = field(default=0)
    x0: float = field(default=0)
    dt: float = field(default=0.1)

    def __post_init__(self):
        self._validate_initial_state()

    def _validate_initial_state(self) -> None:
        if self.derivative_function is None:
            raise ValueError('Derivative function must be provided at initialization')
        elif not callable(self.derivative_function):
            raise ValueError(f'Derivative function must be callable. Got: {self.derivative_function}')

        derivative_function_signature = inspect.signature(self.derivative_function)
        non_default_args = [
            p for p in derivative_function_signature.parameters.values()
            if p.default is inspect.Parameter.empty and
               p.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        ]
        if len(non_default_args) != 1:
            raise ValueError(f'Derivative function must take exactly one positional argument. '
                             f'Got: {non_default_args}. arguments: {derivative_function_signature.parameters}')

        if self.x0 is None:
            raise ValueError('Initial state must include x0')
        if self.dt is None:
            raise ValueError('Initial state must include dt')
        if self.t0 is None:
            raise ValueError('Initial state must include t0')

    @abstractmethod
    def step(self) -> Dict[str, Any]:
        """
        Perform one step of the numerical method.
        Must be implemented by concrete subclasses.
        """
        pass