from typing import Dict

from ODE.FirstOrder.FirstOrderODE import FirstOrderODE
from StopConditions.StopIfNaN import StopIfNaN


class FirstOrderODEHeunMethod(FirstOrderODE):

    def __post_init__(self):
        super().__post_init__()
        self.add_stop_condition(StopIfNaN(track_variables=['x','x_p','t','dx_dt']))

    @property
    def initial_state(self) -> dict:
        return dict(
            x=self.x0, x_p=self.x0, t=self.t0, dt=self.dt,
            dx_dt=self.derivative_function(self.x0, self.t0))

    def step(self) -> Dict[str, float]:
        """
        dx/dt = f(x, t)
        1. Predictor: x*_{n+1} = x_n + dt * f(x_n)
        2. Corrector: x_{n+1} = x_n + 0.5 * dt * (f(x_n, t_n) + f(x*_{n+1}, t_{n+1}))
        :return:
        """

        tn = self.history['t']
        xn = self.history['x']
        dt = self.history['dt']

        tnp1 = tn + dt

        dx_dt = self.derivative_function(xn, tn)
        x_predictor = xn + dt * dx_dt

        dx_dt_predictor = self.derivative_function(x_predictor, tnp1)
        xnp1 = xn + 0.5 * dt * (dx_dt + dx_dt_predictor)

        return dict(x=xnp1, x_p=x_predictor, t=tnp1, dt=dt, dx_dt=dx_dt)


