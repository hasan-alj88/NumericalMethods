from dataclasses import field, dataclass

import sympy

from FindRoots.FindOneRoot import FindOneRoot
from LogAndReraiseException import LogAndReraise
from log_config import get_logger

logger = get_logger(__name__)

@dataclass
class NewtonRaphsonMethod(FindOneRoot):
    x0: float = field(default=0)
    derivative_function: callable = field(default=None, init=False)
    function_sym: sympy.Function = field(default=None, init=False)
    derivative_sym: sympy.Function = field(default=None, init=False)

    def initialize(self) -> None:
        self._single_argument_function(self.function)
        x_sym = sympy.Symbol('x')

        with LogAndReraise(
            exception_type=ValueError,
            log_message="Could not symbolically differentiate.",
            message="Please provide a derivative function manually or use "
                    "sympy functions instead of numpy or other pakages functions.",
            logger=logger
        ):
            self.function_sym = self.function(x_sym)
            self.derivative_sym = sympy.diff(self.function_sym, x_sym)
            self.derivative_function = lambda x_in :self.derivative_sym.evalf(subs={x_sym: x_in})

        logger.info(f'f\'(x) = {self.derivative_sym}')
        self.initial_state = dict(x_root=self.x0, fx_root=self.function(self.x0))
        super().initialize()

    def step(self) -> dict:
        """
        x_n+1 = x_n - f(x_n)/f'(x_n)'
        """
        x_old = self.history.loc[self.last_iteration, 'x_root']
        x_new = x_old - self.function(x_old) / self.derivative_function(x_old)
        self.x_root = x_new
        logger.info(f'x_root = {x_new:0.3e}')
        return dict(x_root=x_new, fx_root=self.function(x_new))
