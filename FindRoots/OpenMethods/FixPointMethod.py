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
    The function f(t) = 0 = g(t) - t -> g(t) = t.
    t_n+1 = g(t_n)
    :ivar t0: Initial guess for the root in the iteration process.
    :type t0: float
    """
    t0 : float = field(default=None)

    def __post_init__(self):
        if self.t0 is None:
            raise ValueError("t0 cannot be None")
        self.add_stop_condition(StopAtPlateau(tracking='t', patience=self.patience,
                                              absolute_tolerance=self.absolute_tolerance,
                                              relative_tolerance=self.relative_tolerance))
        self.add_stop_condition(StopIfNaN(track_variables=['g','t']))

    @property
    def initial_state(self) -> dict:
        return dict(t=self.t0, g=self.function(self.t0))


    def step(self) -> dict:
        """
        Fixed point iteration with relaxation:
        t_{n+1} = g(t_n)
        """
        t_old = self.history['t']
        t_new = self.function(t_old)
        logger.info(f't_root = {t_new:0.3e}')
        return dict(t=t_new, g=self.function(t_new))

    def plot_function(self,
                      t_min: float, t_max: float,
                      ax: plt.Axes = None, resolution: int = 1000, *args,
                      **kwargs) -> plt.Axes:
        # plot the function
        ax = super().plot_function(t_min, t_max, ax, resolution, label='g(t)')

        # Plot the identity line y=t
        ax.plot(
            np.linspace(t_min, t_max, 2),
            np.linspace(t_min, t_max, 2), color='black', linewidth=1.5, label='y=t')

        # Get the t values from history
        t_values = self.history.to_data_frame['t'].values

        # Plot first point on x=t line
        ax.scatter(t_values[0], t_values[0], color='blue', marker='o')

        # Convergence pattern
        t_current = t_values[0]
        for t_next in t_values[1:]:
            f_t_current = self.function(t_current)

            # Draw vertical line from (t_current, t_current) to (t_current, f_t_current)
            ax.plot([t_current, t_current], [t_current, f_t_current], 'r-')

            # Draw horizontal line from (t_current, f_t_current) to (t_next, f_t_current)
            ax.plot([t_current, t_next], [f_t_current, f_t_current], 'r-')

            # Plot the point at (t_current, f_t_current)
            ax.scatter(t_current, f_t_current, color='green', marker='o')

            # Plot the point at (t_next, f_t_current)
            ax.scatter(t_next, f_t_current, color='blue', marker='o')

            # Update t_current for next iteration
            t_current = t_next

        plt.legend()
        return ax