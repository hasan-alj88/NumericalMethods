from dataclasses import dataclass
from typing import ClassVar, Set, Dict, Any

from ODE.ExplicitFirstOrderODE.FirstOrderAutonomousODE import FirstOrderAutonomousODE


@dataclass
class FirstOrderAutonomousODEEulerMethod(FirstOrderAutonomousODE):
    """
    Implements the Explicit Euler method (Forward Euler) for solving Explict first-order linear ODEs.
    dx/dt = f(x)
    Uses the formula: x_{n+1} = x_n + dt * f(x_n)
    """
    variables: ClassVar[Set[str]] = {'t', 'x', 'dt', 'dx_dt'}

    def initialize(self) -> None:
        dx_dt_0 = self.derivative_function(self.x0)
        self.initial_state = dict(
            t=self.t0,
            x=self.x0,
            dt=self.dt,
            dx_dt=dx_dt_0
        )
        super().initialize()

    def step(self) -> Dict[str, Any]:
        t_n = self.history.loc[self.last_iteration, 't']
        x_n = self.history.loc[self.last_iteration, 'x']
        dt = self.history.loc[self.last_iteration, 'dt']

        dx_dt_n = self.derivative_function(x_n)

        t_n_plus_1 = t_n + dt
        x_n_plus_1 = x_n + dt * dx_dt_n

        return dict(
            t=t_n_plus_1,
            x=x_n_plus_1,
            dt=dt,
            dx_dt=dx_dt_n
        )