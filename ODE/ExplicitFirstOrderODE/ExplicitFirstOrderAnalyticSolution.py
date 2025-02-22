from idlelib.history import History
from typing import Dict, Any

from Numerical import HistoryType
from ODE.ExplicitFirstOrderODE.ExplicitFirstOrderODE import ExplicitFirstOrderMethod


class ExplicitFirstOrderAnalyticSolution(ExplicitFirstOrderMethod):
    analytic_solution: callable


    def initialize(self) -> None:
        self._single_argument_function(self.analytic_solution)
        dx_dt_0 = self.derivative_function(self.x0)
        self.initial_state = dict(
            t=self.t0,
            x=self.x0,
            dt=self.dt,
            dx_dt=dx_dt_0
        )
        super().initialize()


    def step(self) -> Dict[str, Any]:
        t_n = self.history['t'][self.last_iteration]
        x_n = self.history['x'][self.last_iteration]
        dt = self.history['dt'][self.last_iteration]

        dx_dt_n = self.derivative_function(x_n)

        t_n_plus_1 = t_n + dt
        x_n_plus_1 = self.analytic_solution(t_n_plus_1)

        return dict(
            t=t_n_plus_1,
            x=x_n_plus_1,
            dt=dt,
            dx_dt=dx_dt_n
        )


    def numerical_solution_error(self, numeric_solution: HistoryType) -> HistoryType:
        t = set(numeric_solution['t']).intersection(set(self.history['t']))
        raise NotImplementedError

