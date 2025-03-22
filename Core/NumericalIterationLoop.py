from decimal import Decimal, localcontext
from inspect import signature
from typing import Dict, Callable, List, TypeVar, Generic, Type, Tuple, Iterator

from pydantic.v1 import BaseModel, Field, validator

# Define a TypeVar for the state model
StateType = TypeVar('StateType', bound=BaseModel)
StopConditionType = TypeVar('StopConditionType', bound=BaseModel)


class NumericalIterationLoop(BaseModel, Generic[StateType]):
    initial_state: StateType
    step: Callable[[List[StateType]], StateType]
    absolute_tolerance: Decimal = Decimal(1e-10)
    relative_tolerance: Decimal = Decimal(1e-6)
    max_iterations: int = 1_000
    stop_conditions: List[StopConditionType] = Field(default_factory=list)
    method_name: str = "Numerical Iteration"
    history: List[StateType] = Field(default_factory=list, init=False)
    stop_condition_history: List[Dict[str, Tuple[bool, str]]] = Field(default_factory=list, init=False)
    _current_iteration: int = Field(default=0, init=False)
    _state_model: Type[BaseModel] = Field(default=None, init=False)

    @validator('initial_state')
    def set_state_model(cls, v):
        return v

    @validator('step')
    def validate_step_function(cls, v, values):
        if 'initial_state' in values:
            # Store the type of the initial state
            values['_state_model'] = type(values['initial_state'])
        return v

    @validator('stop_conditions')
    def validate_stop_conditions(cls, v, values):
        # Validate all stop condition items have an is_stop method
        # that takes List[StateType] as argument and returns bool or Tuple[bool, str]
        for i, condition in enumerate(v):
            if not hasattr(condition, 'is_stop'):
                raise ValueError(f"Stop condition at index {i} must have an 'is_stop' method")

            # Verify the method signature
            sig = signature(condition.is_stop)
            if len(sig.parameters) != 1:
                raise ValueError(f"is_stop method of stop condition at index {i} must take exactly one argument")

            # We can't validate the exact return type at this point
        return v

    @validator('history', always=True)
    def validate_history(cls, v, values):
        # This will validate during runtime as history gets populated
        return v

    def add_to_history(self, state: StateType):
        # Runtime validation that the state is of the correct type
        if not isinstance(state, type(self.initial_state)):
            raise TypeError(f"State must be of type {type(self.initial_state).__name__}")
        self.history.append(state)

    @property
    def precision(self) -> int:
        p = max(
            self.absolute_tolerance.as_tuple().exponent,
            self.relative_tolerance.as_tuple().exponent
        )
        return abs(p) + 1

    def __repr__(self) -> str:
        stop_conditions_repr = ", ".join(condition.__class__.__name__ for condition in self.stop_conditions)

        return (
            f"{self.__class__.__name__}("
            f"method_name='{self.method_name}', "
            f"state_type={self._state_model.__name__ if self._state_model else 'Unknown'}, "
            f"iterations={self._current_iteration}/{self.max_iterations}, "
            f"abs_tol={self.absolute_tolerance}, "
            f"rel_tol={self.relative_tolerance}, "
            f"precision={self.precision}, "
            f"stop_conditions=[{stop_conditions_repr}])"
        )

    def __str__(self) -> str:
        context_info = f"Decimal precision: {self.precision} digits"
        tolerance_info = f"Tolerances: abs={self.absolute_tolerance}, rel={self.relative_tolerance}"

        stop_conditions_info = "Stop conditions: "
        if not self.stop_conditions:
            stop_conditions_info += "None"
        else:
            stop_conditions_info += ", ".join(condition.__class__.__name__ for condition in self.stop_conditions)

        iteration_info = f"Iterations: {self._current_iteration}/{self.max_iterations}"

        return (
            f"{self.method_name} Iteration Loop\n"
            f"State type: {self._state_model.__name__ if self._state_model else 'Unknown'}\n"
            f"{context_info}\n"
            f"{tolerance_info}\n"
            f"{stop_conditions_info}\n"
            f"{iteration_info}"
        )

    def __iter__(self) -> Iterator[Tuple[StateType, Dict[str, Tuple[bool, str]]]]:
        """Iterator protocol implementation that yields each iteration state and stop condition results."""
        # Reset history if we're running again
        self.history = []
        self.stop_condition_history = []

        # Add the initial state
        self.add_to_history(self.initial_state)
        self._current_iteration = 0

        for iteration in range(1, self.max_iterations + 1):
            with localcontext() as ctx:
                ctx.prec = self.precision
                new_state = self.step(self.history)
                self.add_to_history(new_state)
                self._current_iteration = iteration

            # Process stop conditions, which now return (bool, str) or just bool
            current_stop_conditions = {}
            for stop_condition in self.stop_conditions:
                result = stop_condition.is_stop(self.history)

                # Handle different return types
                if isinstance(result, bool):
                    # If just a boolean is returned, add empty log
                    is_stop = result
                    log = ""
                elif isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], bool) and isinstance(
                        result[1], str):
                    # If (bool, str) tuple is returned
                    is_stop, log = result
                else:
                    raise TypeError(
                        f"is_stop method of {stop_condition.__class__.__name__} must return "
                        f"either a boolean or a tuple of (boolean, string). Got {type(result)}"
                    )

                current_stop_conditions[stop_condition.__class__.__name__] = (is_stop, log)

            self.stop_condition_history.append(current_stop_conditions)
            yield new_state, current_stop_conditions

            # Check if all stop conditions are met
            if all(condition[0] for condition in current_stop_conditions.values()):
                break

    def run(self) -> Tuple[StateType, Dict[str, Tuple[bool, str]]]:
        """
        Run the iteration loop until completion and return the final state and stop conditions.
        """
        last_state = None
        last_conditions = {}

        for state, conditions in self:
            last_state = state
            last_conditions = conditions

        return last_state, last_conditions