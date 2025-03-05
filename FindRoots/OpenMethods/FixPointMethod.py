from dataclasses import field, dataclass

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
    relaxation_factor : float = field(default=1.0)

    def __post_init__(self):
        if self.t0 is None:
            raise ValueError("t0 cannot be None")
        self.add_stop_condition(StopAtPlateau(tracking='t', patience=self.patience, tolerance=self.tolerance))
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






