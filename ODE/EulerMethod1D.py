import inspect
from dataclasses import dataclass, field
from typing import Dict, Any, ClassVar, Set, Callable
from Numerical import Numerical


@dataclass
class EulerMethod1D(Numerical):
    derivative_function: Callable = field(default=None)
    t0: float = field(default=0)
    x0: float = field(default=0)
    dt: float = field(default=0.1)
    variables: ClassVar[Set[str]] = {'t', 'x', 'dt', 'dx_dt'}

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

    def initialize(self) -> None:
        # Calculate initial derivative
        dx_dt_0 = self.derivative_function(self.x0)

        self.initial_state = dict(
            t=self.t0,
            x=self.x0,
            dt=self.dt,
            dx_dt=dx_dt_0
        )
        super().initialize()

    def step(self) -> Dict[str, Any]:
        """
        Perform one step of Euler's method for ODE:
        x_{n+1} = x_n + dt * dx_dt_n

        Where dx_dt_n is calculated from the current state x_n

        :return: New state containing t, x, dt, dx_dt
        """
        # Get current values
        t_n = self.history['t'][self.last_iteration]
        x_n = self.history['x'][self.last_iteration]
        dt = self.history['dt'][self.last_iteration]

        # Calculate derivative at current state
        dx_dt_n = self.derivative_function(x_n)

        # Calculate new state using Euler's method
        t_n_plus_1 = t_n + dt
        x_n_plus_1 = x_n + dt * dx_dt_n

        # Return new state
        return dict(
            t=t_n_plus_1,
            x=x_n_plus_1,
            dt=dt,
            dx_dt=dx_dt_n
        )