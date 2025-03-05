from typing import Dict

from ODE.FirstOrder.FirstOrderODE import FirstOrderODE
from StopConditions.StopIfNaN import StopIfNaN


class FirstOrderODEEulerMethod(FirstOrderODE):
    def __post_init__(self):
        super().__post_init__()
        self.add_stop_condition(StopIfNaN(track_variables=['x' ,'t','dx_dt']))


    @property
    def initial_state(self) -> dict:
        return dict(x=self.x0, t=self.t0, dt=self.dt, dx_dt=self.derivative_function(self.x0, self.t0))

    def step(self) -> Dict[str, float]:
        """
        x_{n+1} = x_n + dt * f(x_n, t_n)
        """
        t_n = self.history['t']
        x_n = self.history['x']
        dt = self.history['dt']
        
        t_np1 = t_n + dt
        dx_dt = self.derivative_function(x_n, t_n)
        x_np1 = x_n + dt * dx_dt

        return dict(x=x_np1, t=t_np1, dt=dt, dx_dt=dx_dt)
