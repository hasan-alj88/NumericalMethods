from dataclasses import field, dataclass
from typing import Tuple

import numpy as np
import sympy
from matplotlib import pyplot as plt

from FindRoots.RootFinder import RootFinder
from StopConditions.StopIfEqual import StopIfZero
from StopConditions.StopIfNaN import StopIfNaN
from utils.ExceptionTools import LogAndReraise
from utils.ValidationTools import function_arg_count

@dataclass
class NewtonRaphsonMethod(RootFinder):
    t0: float = field(default=0)
    derivative_function: callable = field(default=None, init=False)
    function_sym: sympy.Function = field(default=None, init=False)
    derivative_sym: sympy.Function = field(default=None, init=False)

    def __post_init__(self) -> None:
        if function_arg_count(self.function) != 1:
            raise ValueError(f'Function must take 1 argument, not {function_arg_count(self.function)}')


        with LogAndReraise(
                logger=self.logger,
                message=f'The Function must be differentiable. '
                        f'Use SymPy functions.'):
            t_sym = sympy.Symbol('t')
            self.function_sym = self.function(t_sym)
            self.logger.info(f'f(t) = {self.function_sym}')
            self.derivative_sym = sympy.diff(self.function_sym, t_sym)
            self.derivative_function = lambda t_n :self.derivative_sym.evalf(subs={t_sym: t_n})
            self.logger.info(f'f\'(t) = {self.derivative_sym}')

        self.add_stop_condition(StopIfZero(tracking='f', patience=self.patience,
                                           absolute_tolerance=self.absolute_tolerance,
                                           relative_tolerance=self.relative_tolerance))
        self.add_stop_condition(StopIfZero(tracking='df_dt', patience=0,
                                           absolute_tolerance=self.absolute_tolerance,
                                           relative_tolerance=self.relative_tolerance))
        self.add_stop_condition(StopIfNaN(track_variables=['f','t', 'df_dt']))

    @property
    def initial_state(self) -> dict:
        return dict(
            t=self.t0,
            f=self.function(self.t0),
            df_dt=self.derivative_function(self.t0)
        )


    def step(self) -> dict:
        """
        t_n+1 = t_n - f(t_n)/f'(t_n)'
        """
        t_n = self.history['t']
        f = self.function(t_n)
        fp = self.derivative_function(t_n)
        t_np1 = t_n - f / fp
        self.logger.info(f't_root = {t_np1:0.3e}')
        return dict(
            t=t_np1,
            f=self.function(t_np1),
            df_dt=fp,
        )

    def plot_tangent(self, step: int, ax: plt.Axes) -> plt.Axes:
        """
        Plots the tangent line used in the Newton-Raphson method at a specific step.

        Args:
            step: The iteration step to plot
            ax: The matplotlib axes to plot on

        Returns:
            The matplotlib axes with the tangent line added
        """
        # Get points
        t_n = float(self.history(step, 't'))
        t_np1 = float(self.history(step + 1, 't'))
        f_n = float(self.history(step, 'f'))

        ax.plot([t_n, t_np1], [f_n, 0], color='red', linestyle='-.')

        # Add the text label for the tangent line
        ax.text(
            (t_n + t_np1) / 2, f_n / 2,
            f'Tangent Line at $t_{{{step}}}$',
            ha='center', va='center'
        )
        return ax

    def plot_step_point(self, step:int, ax:plt.Axes,
                        x_offset:float=0.1, y_offset:float=0.1) -> plt.Axes:
        t_n = float(self.history(step, 't'))
        f_n = float(self.history(step, 'f'))
        ax.scatter([t_n, t_n], [f_n, 0], color='black', marker='o')
        ax.text(t_n+x_offset, f_n+y_offset, f'$f(t_{{{step}}})$', ha='center', va='center')
        ax.text(t_n-x_offset, y_offset, f'$t_{{{step}}}$', ha='center', va='center')
        ax.plot([t_n, t_n], [f_n, 0], color='blue', linestyle=':')
        return ax

    def plot_function(self, t_min:float, t_max:float, ax:plt.Axes = None, resolution:int=1000, *args, **kwargs) -> plt.Axes:

        if ax is None:
            fig, ax = plt.subplots()

        t_float = np.linspace(t_min, t_max, resolution)
        t = sympy.Symbol('t')
        func = self.function(t)
        f = sympy.lambdify(t, expr=func)
        ax.plot(
            t_float, f(t_float),
            color='black',
            linewidth=1.5,
        )


        ax.set_xlabel('t')
        ax.set_ylabel('f(t)')
        ax.grid(True)
        ax.set_title(f'$f(t)={sympy.latex(self.function(t))}$')

        # Highlight x-Axis
        ax.axhline(y=0, color='k', linewidth=2.0)

        return ax


