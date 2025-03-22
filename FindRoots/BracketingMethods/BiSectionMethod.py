from dataclasses import dataclass
from decimal import Decimal
from functools import partial
from typing import Callable, List

import numpy as np
from pydantic.v1 import BaseModel, Field, validator

from Core.NumericalIterationLoop import NumericalIterationLoop, StopConditionType
from FindRoots.BracketingMethods.BracketingSate import BracketingState
from utils.ValidationTools import is_nan, function_arg_count
from utils.log_config import get_logger

logger = get_logger(__name__)


@dataclass
class BiSectionMethod(BaseModel):
    a0: Decimal = Field(init= True, help="Lower bound of bracketing interval")
    b0: Decimal = Field(init= True, help="Upper bound of bracketing interval")
    function: Callable[[Decimal], Decimal] = Field(init=True, help="f(x)=0")

    patience: int = Field(default=10, help="Number of iterations to wait before stopping", ge=1)
    abs_tol: Decimal = Field(default=Decimal(1e-6), help="Absolute tolerance used for stopping condition", ge=0)
    rel_tol: Decimal = Field(default=Decimal(1e-6), help="Relative tolerance used for stopping condition", ge=0)
    stop_conditions: List[StopConditionType] = Field(default_factory=list, help="List of stop conditions")
    loop: NumericalIterationLoop[BracketingState] = Field(init=False, help="Numerical iteration loop")
    
    @validator('a0', 'b0', 'function', pre=True)
    def validate_bounds(cls, v, values):
        a0 = values.get('a0')
        b0 = values.get('b0')
        function = values.get('function')

        if any([a0, b0, function]) is None:
            raise ValueError("All three of 'a0', 'b0', and 'function' must be provided."
                             f"Got {a0=}, {b0=}, function={function.__name__}")

        if a0 > b0:
            raise ValueError(f"'a0' must be less than 'b0'. Got {a0=} and {b0=}.")

        if function_arg_count(function) != 1:
            raise ValueError(
                f"Function must take exactly one argument. "
                f"Got {function.__name__} with {function_arg_count(function)} arguments."
            )

        fa, fb = function(a0), function(b0)
        if any([is_nan(fa), is_nan(fb)]):
            raise ValueError(f"Function is not defined at bounds. f({a0}) = {fa}, f({b0}) = {fb}")
        if fa * fb > 0:
            raise ValueError(f"Function must have opposite signs at bounds. f({a0}) = {fa}, f({b0}) = {fb}")
        return v

    @validator('stop_conditions', pre=True)
    def validate_stop_conditions(cls, v, values):
        for sc in v:
            if not isinstance(sc, BaseModel):
                raise ValueError(f"All stop conditions must be subclasses of StopConditionType. Got {sc}")
            if not hasattr(sc, 'is_stop'):
                raise ValueError(f"All stop conditions must have an 'is_stop' method.")

    @validator('loop')
    def validate_loop(cls, v, values):
        loop = NumericalIterationLoop(
            initial_state=cls.initial_state,
            step=lambda state: cls.step(state),
            absolute_tolerance=v.get('abs_tol'),
            relative_tolerance=v.get('rel_tol'),
            stop_conditions=v.get('stop_')
        )

    @property
    def initial_state(self) -> BracketingState:
        return BracketingState(
            a=self.a0,
            b=self.b0,
            fa=self.function(self.a0),
            fb=self.function(self.b0),
            root=((self.a0 + self.b0) / 2),
            log='Initial state'
        )

    def step(self, history: List[BracketingState]) -> BracketingState:
        """
        Perform one iteration of the bisection method.

        Returns:
            dict: State variables for current iteration
        """
        # Get previous values
        a, b, fa, fb, root = (
            history[-1].a,
            history[-1].b,
            history[-1].fa,
            history[-1].fb,
            history[-1].root
        )

        fr = self.function(root)

        if fr * fa < 0:
            return BracketingState(
                a=a,
                b=root,
                fa=fa,
                fb=fr,
                root=((a + root) / 2),
                log='Root in Lower bound'
            )
        elif fr * fb < 0:
            return BracketingState(
                a=root,
                b=b,
                fa=fr,
                fb=fb,
                root=((root + b) / 2),
                log='Root in Upper bound'
            )
        elif np.isclose(fr, 0, atol=self.abs_tol, rtol=self.rel_tol):
            return BracketingState(
                a=root,
                b=root,
                fa=fr,
                fb=fr,
                root=root,
                log='Root found at midpoint'
            )
        elif np.isclose(fa, 0, atol=self.abs_tol, rtol=self.rel_tol):
            return BracketingState(
                a=a,
                b=a,
                fa=fa,
                fb=fa,
                root=a,
                log='Root edge case: fa=0'
            )
        elif np.isclose(fb, 0, atol=self.abs_tol, rtol=self.rel_tol):
            return BracketingState(
                a=b,
                b=b,
                fa=fb,
                fb=fb,
                root=b,
                log='Root edge case: fb=0'
            )
        else:
            raise ValueError("Bisection method failed to converge")