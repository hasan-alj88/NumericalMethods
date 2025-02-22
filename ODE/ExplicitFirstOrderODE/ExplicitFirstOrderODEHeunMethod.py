from typing import ClassVar, Set, Dict, Any

from ODE.ExplicitFirstOrderODE.ExplicitFirstOrderODE import ExplicitFirstOrderMethod


class ExplicitFirstOrderODEHeunMethod(ExplicitFirstOrderMethod):
    """
    Implements the Explicit Heun method (Improved Euler) for solving ODEs.
    Uses predictor-corrector approach:
    1. Predictor: x̃_{n+1} = x_n + dt * f(x_n)
    2. Corrector: x_{n+1} = x_n + 0.5 * dt * (f(x_n) + f(x̃_{n+1}))
    """
    variables: ClassVar[Set[str]] = {'t', 'x', 'dt', 'dx_dt', 'predictor_x'}

    def initialize(self) -> None:
        dx_dt_0 = self.derivative_function(self.x0)
        predictor_x0 = self.x0 + self.dt * dx_dt_0

        self.initial_state = dict(
            t=self.t0,
            x=self.x0,
            dt=self.dt,
            dx_dt=dx_dt_0,
            predictor_x=predictor_x0
        )
        super().initialize()

    def step(self) -> Dict[str, Any]:
        t_n = self.history.loc[self.last_iteration, 't']
        x_n = self.history.loc[self.last_iteration, 'x']
        dt = self.history.loc[self.last_iteration, 'dt']

        dx_dt_n = self.derivative_function(x_n)

        # Predictor step
        predictor_x_n_plus_1 = x_n + dt * dx_dt_n
        dx_dt_predictor = self.derivative_function(predictor_x_n_plus_1)

        # Corrector step
        t_n_plus_1 = t_n + dt
        x_n_plus_1 = x_n + 0.5 * dt * (dx_dt_n + dx_dt_predictor)
        dx_dt_n_plus_1 = self.derivative_function(x_n_plus_1)

        return dict(
            t=t_n_plus_1,
            x=x_n_plus_1,
            dt=dt,
            dx_dt=dx_dt_n_plus_1,
            predictor_x=predictor_x_n_plus_1
        )