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
            t_lower=self.t_lower0,
            t_upper=self.t_upper0,
            t_root=(self.t_upper0 + self.t_lower0) / 2.0,
            f_lower=self.function(self.t_lower0),
            f_root=self.function((self.t_upper0 + self.t_lower0) / 2.0),
            f_upper=self.function(self.t_upper0),
            bracket_size=abs(self.t_upper0 - self.t_lower0),
            log='Initial state'
        )

    def _validate_initial_state(self) -> None:
        # Validate that function has opposite signs at bounds
        f_lower = self.function(self.t_lower0)
        f_upper = self.function(self.t_upper0)

        if any([is_nan(b) for b in [f_lower, f_upper]]):
            raise ValueError(f"Function is not defined at bracket endpoints. "
                             f"f({self.t_lower0}) = {f_lower}, f({self.t_upper0}) = {f_upper}")
        elif f_lower * f_upper > 0:
            raise ValueError(
                f"Function must have opposite signs at bracket endpoints. "
                f"f({self.t_lower0}) = {f_lower}, f({self.t_upper0}) = {f_upper}"
            )

    def step(self) -> dict:
        """
        Perform one iteration of the bisection method.

        Returns:
            dict: State variables for current iteration
        """
        # Get previous values
        t_upper = self.history['t_upper']
        t_lower = self.history['t_lower']
        t_root = self.history['t_root']

        # Calculate function values
        f_upper = self.function(t_upper)
        f_lower = self.function(t_lower)
        f_root = self.function(t_root)

        # Check for undefined function values
        for t, f in [(t_lower, f_lower), (t_upper, f_upper), (t_root, f_root)]:
            if is_nan(f):
                raise ValueError(f"The function is not defined at x = {t:0.3e}")

        # Check if we found the root exactly
        if f_lower == 0:
            t_lower_new, t_upper_new = t_lower, t_lower
            log = 'Root found at t_lower'
        elif f_upper == 0:
            t_lower_new, t_upper_new = t_upper, t_upper
            log = 'Root found at t_upper'
        elif f_root == 0:
            t_lower_new, t_upper_new = t_root, t_root
            log = 'Root found at t_root'
        else:
            # Determine which half contains the root
            if f_lower * f_root < 0:
                t_lower_new, t_upper_new = t_lower, t_root
                log = 'Root is in lower half of interval'
            else:
                t_lower_new, t_upper_new = t_root, t_upper
                log = 'Root is in upper half of interval'

        # Calculate new root approximation
        t_root_new = (t_lower_new + t_upper_new) / 2
        # Log current state
        logger.info(log)
        logger.info(f'f({t_root_new:0.3e}) = {f_root:0.3e}')

        # Return new state
        return dict(
            t_lower=t_lower_new,
            t_root=t_root_new,
            t_upper=t_upper_new,
            f_lower=f_lower,
            f_root=f_root,
            f_upper=f_upper,
            bracket_size=abs(t_upper_new - t_lower_new),
            log=log
        )