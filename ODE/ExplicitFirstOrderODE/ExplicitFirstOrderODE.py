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
        self._single_argument_function(self.derivative_function)

        if self.x0 is None:
            raise ValueError('Initial state must include x0')
        if self.dt is None:
            raise ValueError('Initial state must include dt')
        if self.t0 is None:
            raise ValueError('Initial state must include t0')

    @staticmethod
    def _single_argument_function(func):
        if func is None:
            raise ValueError('Derivative function must be provided at initialization')
        elif not callable(func):
            raise ValueError(f'Derivative function must be callable. Got: {type(func)}')

        derivative_function_signature = inspect.signature(func)
        non_default_args = [
            p for p in derivative_function_signature.parameters.values()
            if p.default is inspect.Parameter.empty and
               p.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        ]
        if len(non_default_args) != 1:
            raise ValueError(f'Derivative function must take exactly one positional argument. '
                             f'Got: {non_default_args}. arguments: {derivative_function_signature.parameters}')


    @abstractmethod
    def step(self) -> Dict[str, Any]:
        """
        Perform one step of the numerical method.
        Must be implemented by concrete subclasses.
        """
        pass