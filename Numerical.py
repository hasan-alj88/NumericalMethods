import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Generator, Optional
import matplotlib.pyplot as plt

from log_config import get_logger

logger = get_logger(__name__)


HistoryType = Dict[str, Dict[int, Any]]
StopConditionType = Generator[bool, None, None]


@dataclass
class StopCondition(ABC):
    history: HistoryType = field(default_factory=dict)
    stop_reason: Optional[str] = field(default=None, init=False)
    _generator: Optional[Generator[bool, None, None]] = field(default=None, init=False)

    def __post_init__(self):
        if self.history is None:
            raise ValueError("History cannot be None")

    @abstractmethod
    def stop_condition_generator(self) -> Generator[bool, None, None]:
        """Create a generator that will yield stop condition results"""
        pass

    def next(self, history: HistoryType) -> bool:
        """
         Get next evaluation of the stop condition with updated history.
        :param history: Current computation history.
        :return: True if the stop condition is met (generator raised StopIteration),
          False otherwise
        """
        self.history = history

        # Initialize generator if needed
        if self._generator is None:
            self._generator = self.stop_condition_generator()

        # Get next value
        try:
            next(self._generator)
            return False
        except StopIteration:
            # Reinitialize the generator and try again
            self._generator = self.stop_condition_generator()
            # Stop Condition is met
            return True


@dataclass
class Numerical(ABC):
    initial_state: Dict[str, Any]
    stop_conditions: List[StopCondition] = field(default_factory=list)
    max_iterations: int = 1_000
    verbose: bool = False
    history: HistoryType = field(default_factory=lambda: defaultdict(dict))
    _iteration: int = field(default=0, init=False)

    def add_stop_condition(self, stop_condition: StopCondition) -> None:
        """Add a stop condition to the list of stop conditions"""
        self.stop_conditions.append(stop_condition)

    @abstractmethod
    def initialize(self) -> None:
        self.history.clear()
        self._iteration = 0
        self.record_state(self.initial_state)

    @abstractmethod
    def _validate_initial_state(self) -> None:
        pass

    @abstractmethod
    def step(self) -> Dict[str, Any]:
        pass

    def record_state(self, state: Dict[str, Any]) -> None:
        """Record the current state in history"""
        for var_name, value in state.items():
            self.history[var_name][self._iteration] = value

    def log(self, message: str, verbose: bool = False) -> None:
        """Print a message if verbose is True"""
        logger.info(message)
        if self.verbose or verbose:
            print(message)

    def _check_stop_conditions(self) -> Generator[bool, None, None]:
        """
        Checks if any of the specified stopping conditions are met and raises StopIteration
        if so.
        :yields: Always yields True until a stop condition is met
        """
        for iteration in range(self.max_iterations):
            self._iteration = iteration
            # Check custom stop conditions
            stop_reasons = [
                stop_cond.stop_reason or stop_cond.__class__.__name__
                for stop_cond in self.stop_conditions
                if stop_cond.next(self.history)
            ]

            if stop_reasons:
                for reason in stop_reasons:
                    self.log(f"Stopping condition {reason} met")
                raise StopIteration('Stopping condition(s) met:' + '\n'.join(stop_reasons))

            yield True

        # Handle max iterations reached
        self.log(f"Stopping condition max_iterations reached ({self.max_iterations})")
        raise StopIteration(f'Stopping condition met: max_iterations reached ({self.max_iterations})')

    def run(self) -> HistoryType:
        """
        Run the numerical method using a for-loop with the generator method
        that checks stop conditions using StopIteration.
        :returns: Dict[str, Dict[int, Any]]: Complete history of the computation
        """
        self.history.clear()

        self.log(f"Starting {self.__class__.__name__}", True)
        self.initialize()

        # Main iteration loop using the stop checker generator
        for _ in self._check_stop_conditions():
            self.log('-' * 50, True)
            self.log(f"Iteration {self.iteration}", True)
            self.log('-' * 50, True)

            # Execute algorithm step
            state = self.step()
            self.record_state(state)
            self.log(f"Iteration {self.iteration}: \n{json.dumps(state)}")


        return dict(self.history)

    @property
    def iteration(self) -> int:
        """Current iteration number (read-only)"""
        return self._iteration

    @property
    def last_iteration(self) -> int:
        """Maximum number of iterations"""
        return max([max(values.keys()) if values else -1 for values in self.history.values()], default=-1)

    def get_history_at(self, iteration: int) -> Dict[str, Any]:
        """
        Get all variables' values at a specific iteration.

        Args:
            iteration: Iteration number

        Returns:
            Dict[str, Any]: Variable values at specified iteration
        """
        if iteration < 0:
            raise ValueError(f"Iteration must be non-negative, got {iteration}")
        if iteration > self.last_iteration:
            raise ValueError(f"Iteration {iteration} exceeds maximum recorded iteration {self.last_iteration}")

        return {
            var_name: values.get(iteration)
            for var_name, values in self.history.items()
        }

    def __getitem__(self, item):
        """Dictionary-style access to variable history"""
        if item not in self.history:
            raise KeyError(f"Variable '{item}' not found in history")
        return self.history[item]

    def get_last_state(self) -> Dict[str, Any]:
        """Get the last state of the numerical method"""
        return self.get_history_at(self.last_iteration)

    def export_history(self, filepath: str) -> None:
        """
        Export history to a JSON file

        Args:
            filepath: Path to save the history
        """
        # Convert default dict to regular dict for serialization
        history_dict = {k: dict(v) for k, v in self.history.items()}
        with open(filepath, 'w') as file:
            json.dump(history_dict, file, indent=2)

    def plot_history(self, var_name:str, ax: plt.axis, *args, **kwargs) -> plt.axis:
        """
        Plot history of a variable vs iterations.
        :param var_name: name of the variable to plot.
        :param ax: Plot axis
        """
        data = self.history[var_name].values()
        ax.plot(data, *args, **kwargs)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(var_name)
        return ax
