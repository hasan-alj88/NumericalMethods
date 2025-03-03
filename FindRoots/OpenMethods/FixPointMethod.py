from dataclasses import field, dataclass

from FindRoots.RootFinder import RootFinder
from StopConditions.StopIfEqual import StopIfZero
from utils.log_config import get_logger

logger = get_logger(__name__)

@dataclass
class FixPointMethod(RootFinder):
    """
    Represents the Fixed Point Method for finding one root of an equation.
    The function be rearranged as t = f(t) in order to apply the method.
    t_n+1 = f(t_n)
    :ivar t0: Initial guess for the root in the iteration process.
    :type t0: float
    """
    t0 : float = field(default=None)

    def __post_init__(self):
        self.add_stop_condition(StopIfZero(tracking='f', patience=3, tolerance=1e-6))

    @property
    def initial_state(self) -> dict:
        return dict(t=self.t0, f=self.function(self.t0))

    def initialize(self) -> None:
        if self.t0 is None:
            raise ValueError("t0 cannot be None")
        super().initialize()


    def step(self) -> dict:
        t_old = self.history['t']
        t_new = self.function(t_old)
        logger.info(f't_root = {t_new:0.3e}')
        return dict(t=t_new, f=self.function(t_new))






