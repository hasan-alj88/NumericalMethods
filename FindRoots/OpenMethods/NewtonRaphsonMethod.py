from dataclasses import field, dataclass

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
    x0: float = field(default=0)
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
            x_sym = sympy.Symbol('x')
            self.function_sym = self.function(x_sym)
            self.logger.info(f'f(x) = {self.function_sym}')
            self.derivative_sym = sympy.diff(self.function_sym, x_sym)
            self.derivative_function = lambda x_n :self.derivative_sym.evalf(subs={x_sym: x_n})
            self.logger.info(f'f\'(x) = {self.derivative_sym}')

        self.add_stop_condition(StopIfZero(tracking='f', patience=self.patience,
                                           absolute_tolerance=self.absolute_tolerance,
                                           relative_tolerance=self.relative_tolerance))
        self.add_stop_condition(StopIfZero(tracking='df_dx', patience=0,
                                           absolute_tolerance=self.absolute_tolerance,
                                           relative_tolerance=self.relative_tolerance))
        self.add_stop_condition(StopIfNaN(track_variables=['f','x', 'df_dx']))

    @property
    def initial_state(self) -> dict:
        return dict(
            x=self.x0,
            f=self.function(self.x0),
            df_dx=self.derivative_function(self.x0)
        )


    def step(self) -> dict:
        """
        x_n+1 = x_n - f(x_n)/f'(x_n)
        """
        x_n = self.history['x']
        f = self.function(x_n)
        fp = self.derivative_function(x_n)
        x_np1 = x_n - f / fp
        self.logger.info(f't_root = {x_np1:0.3e}')
        return dict(
            x=x_np1,
            f=self.function(x_np1),
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
        x_n = float(self.history(step, 'x'))
        x_np1 = float(self.history(step + 1, 'x'))
        f_n = float(self.history(step, 'f'))

        ax.plot([x_n, x_np1], [f_n, 0], color='red', linestyle='-.')

        # Add the text label for the tangent line
        ax.text(
            (x_n + x_np1) / 2, f_n / 2,
            f'Tangent Line at $t_{{{step}}}$',
            ha='center', va='center'
        )
        return ax

    def plot_step_point(self, step:int, ax:plt.Axes,
                        x_offset:float=0.1, y_offset:float=0.1) -> plt.Axes:
        x_n = float(self.history(step, 'x'))
        f_n = float(self.history(step, 'f'))
        ax.scatter([x_n, x_n], [f_n, 0], color='black', marker='o')
        ax.text(x_n+x_offset, f_n+y_offset, f'$f(x_{{{step}}})$', ha='center', va='center')
        ax.text(x_n-x_offset, y_offset, f'$x_{{{step}}}$', ha='center', va='center')
        ax.plot([x_n, x_n], [f_n, 0], color='blue', linestyle=':')
        return ax

    def plot_function(self, x_min:float, x_max:float, ax:plt.Axes = None, resolution:int=1000, *args, **kwargs) -> plt.Axes:

        if ax is None:
            fig, ax = plt.subplots()

        x_float = np.linspace(x_min, x_max, resolution)
        x = sympy.Symbol('x')
        func = self.function(x)
        f = sympy.lambdify(x, expr=func)
        ax.plot(
            x_float, f(x_float),
            color='black',
            linewidth=1.5,
        )


        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.grid(True)
        ax.set_title(f'$f(x)={sympy.latex(self.function(x))}$')

        # Highlight x-Axis
        ax.axhline(y=0, color='k', linewidth=2.0)

        return ax


