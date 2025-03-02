from dataclasses import field, dataclass

import sympy

from FindRoots.RootFinder import RootFinder
from StopConditions.StopIfEqual import StopIfZero
from utils.ExceptionTools import LogAndReraise
from utils.ValidationTools import function_arg_count
from utils.log_config import get_logger

logger = get_logger(__name__)

@dataclass
class NewtonRaphsonMethod(RootFinder):
    t0: float = field(default=0)
    derivative_function: callable = field(default=None, init=False)
    function_sym: sympy.Function = field(default=None, init=False)
    derivative_sym: sympy.Function = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.add_stop_condition(StopIfZero(tracking='f', patience=3, tolerance=1e-6))
        super().__post_init__()

    @property
    def initial_state(self) -> dict:
        return dict(t=self.t0, f=self.function(self.t0), df_dt=self.derivative_function(self.t0))

    def initialize(self) -> None:
        if function_arg_count(self.function) != 1:
            raise ValueError(f'Function must take 1 argument, not {function_arg_count(self.function)}')


        with LogAndReraise(
                logger=self.logger,
                message=f'The Function must be differentiable. '
                        f'Use SymPy functions.'):
            t_sym = sympy.Symbol('t')
            self.function_sym = self.function(t_sym)
            self.derivative_sym = sympy.diff(self.function_sym, t_sym)
            self.derivative_function = lambda t_n :self.derivative_sym.evalf(subs={t_sym: t_n})


        logger.info(f'f\'(t) = {self.derivative_sym}')
        super().initialize()

    def step(self) -> dict:
        """
        t_n+1 = t_n - f(t_n)/f'(t_n)'
        """
        t_n = self.history['t']
        t_np1 = t_n - self.function(t_n) / self.derivative_function(t_n)
        logger.info(f't_root = {t_np1:0.3e}')
        return dict(
            t=t_np1,
            f=self.function(t_np1),
            df_dt=self.derivative_function(t_np1),
        )