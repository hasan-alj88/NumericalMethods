from typing import Dict

from ODE.FirstOrder.FirstOrderODE import FirstOrderODE


class FirstOrderODEHeunMethod(FirstOrderODE):

    @property
    def initial_state(self) -> dict:
        return dict(
            x_c=0.0, x_p=self.x0, t=self.t0, dt=self.dt,
            dx_dt=self.derivative_function(self.x0, self.t0))

    def step(self) -> Dict[str, float]:
        """
        1. Predictor: x*_{n+1} = x_n + dt * f(x_n)
        2. Corrector: x_{n+1} = x_n + 0.5 * dt * (f(x_n) + f(x*_{n+1}))
        :return:
        """

        t = self.history['t']
        x = self.history['x']
        dt = self.history['dt']

        t_1 = t + dt
        x_predictor = x + dt * self.derivative_function(x, t)
        x_1 = x + 0.5 * dt * (self.derivative_function(x, t) + self.derivative_function(x_predictor, t_1))

        return dict(x_c=x_1, x_p=x_predictor, t=t_1, dt=dt, dx_dt=self.derivative_function(x, t))


