from dataclasses import field, dataclass

import numpy as np
from matplotlib import pyplot as plt

from FindRoots.RootFinder import RootFinder
from StopConditions.StopAtPlateau import StopAtPlateau
from StopConditions.StopIfNaN import StopIfNaN
from utils.log_config import get_logger

logger = get_logger(__name__)

@dataclass
class FixPointMethod(RootFinder):
    """
    Represents the Fixed Point Method for finding one root of an equation.
    The function f(x) = 0 = g(t) - x -> g(x) = x.
    x_n+1 = g(x_n)
    :ivar x0: Initial guess for the root in the iteration process.
    :type x0: float
    """
    x0 : float = field(default=None)

    def __post_init__(self):
        if self.x0 is None:
            raise ValueError("x0 cannot be None")
        self.add_stop_condition(StopAtPlateau(tracking='x', patience=self.patience,
                                              absolute_tolerance=self.absolute_tolerance,
                                              relative_tolerance=self.relative_tolerance))
        self.add_stop_condition(StopIfNaN(track_variables=['g','x']))

    @property
    def initial_state(self) -> dict:
        return dict(t=self.x0, g=self.function(self.x0))


    def step(self) -> dict:
        """
        Fixed point iteration with relaxation:
        x_{n+1} = g(x_n)
        """
        x_old = self.history['x']
        x_new = self.function(x_old)
        logger.info(f'x_root = {x_new:0.3e}')
        return dict(t=x_new, g=self.function(x_new))

    def plot_function(self,
                      x_min: float, x_max: float,
                      ax: plt.Axes = None, resolution: int = 1000, *args,
                      **kwargs) -> plt.Axes:
        # plot the function
        ax = super().plot_function(x_min, x_max, ax, resolution, label='g(t)')

        # Plot the identity line y=t
        ax.plot(
            np.linspace(x_min, x_max, 2),
            np.linspace(x_min, x_max, 2), color='black', linewidth=1.5, label='y=t')

        # Get the t values from history
        x_values = self.history.to_data_frame['x'].values

        # Plot first point on x=t line
        ax.scatter(x_values[0], x_values[0], color='blue', marker='o')

        # Convergence pattern
        x_current = x_values[0]
        for x_next in x_values[1:]:
            f_x_current = self.function(x_current)

            # Draw vertical line from (x_current, x_current) to (x_current, f_x_current)
            ax.plot([x_current, x_current], [x_current, f_x_current], 'r-')

            # Draw horizontal line from (x_current, f_x_current) to (x_next, f_x_current)
            ax.plot([x_current, x_next], [f_x_current, f_x_current], 'r-')

            # Plot the point at (x_current, f_x_current)
            ax.scatter(x_current, f_x_current, color='green', marker='o')

            # Plot the point at (x_next, f_x_current)
            ax.scatter(x_next, f_x_current, color='blue', marker='o')

            # Update x_current for next iteration
            x_current = x_next

        plt.legend()
        return ax