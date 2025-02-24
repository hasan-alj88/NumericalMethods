import numpy as np

from FindRoots.BracketingMethods.BracketingMethods import BracketingMethods
from log_config import get_logger

logger = get_logger(__name__)

class BiSectionMethod(BracketingMethods):

    def initialize(self) -> None:
        """Initialize the method with starting values and validate inputs."""
        self._validate_initial_state()

        # Validate that function has opposite signs at bounds
        fx_lower = self.function(self.x_lower)
        fx_upper = self.function(self.x_upper)

        if fx_lower * fx_upper > 0:
            raise ValueError(
                f"Function must have opposite signs at bracket endpoints. "
                f"f({self.x_lower}) = {fx_lower}, f({self.x_upper}) = {fx_upper}"
            )

        # Calculate initial root approximation
        self.x_root = (self.x_upper + self.x_lower) / 2

        # Set initial state
        self.initial_state = dict(
            x_upper=self.x_upper,
            x_lower=self.x_lower,
            x_root=self.x_root,
            fx_lower=fx_lower,
            fx_upper=fx_upper,
            fx_root=self.function(self.x_root),
            log='Initial state'
        )

        super().initialize()

    def step(self) -> dict:
        """
        Perform one iteration of the bisection method.

        Returns:
            dict: State variables for current iteration
        """
        # Get previous values
        x_upper = self.history.loc[self.last_iteration, 'x_upper']
        x_lower = self.history.loc[self.last_iteration, 'x_lower']
        x_root = self.history.loc[self.last_iteration, 'x_root']

        # Calculate function values
        f_upper = self.function(x_upper)
        f_lower = self.function(x_lower)
        f_root = self.function(x_root)

        # Check for undefined function values
        for x, fx in [(x_lower, f_lower), (x_upper, f_upper), (x_root, f_root)]:
            if np.isnan(fx):
                raise ValueError(f"The function is not defined at x = {x:0.3e}")

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
        self.x_root = x_root_new
        # Log current state
        logger.info(log)
        logger.info(f'f({x_root_new:0.3e}) = {f_root:0.3e}')

        self.self_stopping = self._is_x_root_converged()

        # Return new state
        return dict(
            x_upper=x_upper_new,
            x_lower=x_lower_new,
            x_root=x_root_new,
            fx_lower=self.function(x_lower_new),
            fx_upper=self.function(x_upper_new),
            fx_root=self.function(x_root_new),
            log=log
        )