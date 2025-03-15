from dataclasses import dataclass

from FindRoots.BracketingMethods.BracketingMethods import BracketingMethods
from StopConditions.StopIfEqual import StopIfZero
from StopConditions.StopIfNaN import StopIfNaN
from utils.ValidationTools import is_nan
from utils.log_config import get_logger

logger = get_logger(__name__)


@dataclass
class BiSectionMethod(BracketingMethods):

    def __post_init__(self) -> None:
        self.add_stop_condition(StopIfZero(tracking='f_root', patience=3,
                                           absolute_tolerance=1e-6, relative_tolerance=1e-6))
        self.add_stop_condition(StopIfZero(tracking='bracket_size', patience=1,
                                           absolute_tolerance=1e-6, relative_tolerance=1e-6))
        self.add_stop_condition(StopIfNaN(track_variables=['f_lower', 'f_upper', 'f_root']))

    @property
    def initial_state(self) -> dict:
        return dict(
            x_lower=self.a,
            x_upper=self.b,
            x_root=(self.b + self.a) / 2.0,
            f_lower=self.function(self.a),
            f_root=self.function((self.b + self.a) / 2.0),
            f_upper=self.function(self.b),
            bracket_size=abs(self.b - self.a),
            log='Initial state'
        )

    def _validate_initial_state(self) -> None:
        # Validate that function has opposite signs at bounds
        f_lower = self.function(self.a)
        f_upper = self.function(self.b)

        if any([is_nan(b) for b in [f_lower, f_upper]]):
            raise ValueError(f"Function is not defined at bracket endpoints. "
                             f"f({self.a}) = {f_lower}, f({self.b}) = {f_upper}")
        elif f_lower * f_upper > 0:
            raise ValueError(
                f"Function must have opposite signs at bracket endpoints. "
                f"f({self.a}) = {f_lower}, f({self.b}) = {f_upper}"
            )

    def step(self) -> dict:
        """
        Perform one iteration of the bisection method.

        Returns:
            dict: State variables for current iteration
        """
        # Get previous values
        x_upper = self.history['x_upper']
        x_lower = self.history['x_lower']
        x_root = self.history['x_root']

        # Calculate function values
        f_upper = self.function(x_upper)
        f_lower = self.function(x_lower)
        f_root = self.function(x_root)

        # Check for undefined function values
        for t, f in [(x_lower, f_lower), (x_upper, f_upper), (x_root, f_root)]:
            if is_nan(f):
                raise ValueError(f"The function is not defined at x = {t:0.3e}")

        # Check if we found the root exactly
        if f_lower == 0:
            x_lower_new, x_upper_new = x_lower, x_lower
            log = 'Root found at x_lower'
        elif f_upper == 0:
            x_lower_new, x_upper_new = x_upper, x_upper
            log = 'Root found at x_upper'
        elif f_root == 0:
            x_lower_new, x_upper_new = x_root, x_root
            log = 'Root found at x_root'
        else:
            # Determine which half contains the root
            if f_lower * f_root < 0:
                x_lower_new, x_upper_new = x_lower, x_root
                log = 'Root is in lower half of interval'
            else:
                x_lower_new, x_upper_new = x_root, x_upper
                log = 'Root is in upper half of interval'

        # Calculate new root approximation
        x_root_new = (x_lower_new + x_upper_new) / 2
        # Log current state
        logger.info(log)
        logger.info(f'f({x_root_new:0.3e}) = {f_root:0.3e}')

        # Return new state
        return dict(
            x_lower=x_lower_new,
            x_root=x_root_new,
            x_upper=x_upper_new,
            f_lower=f_lower,
            f_root=f_root,
            f_upper=f_upper,
            bracket_size=abs(x_upper_new - x_lower_new),
            log=log
        )