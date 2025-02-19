import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Generator, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from log_config import get_logger

logger = get_logger(__name__)


HistoryType = Dict[str, Dict[int, Any]]
StopConditionType = Generator[bool, None, None]


@dataclass
class StopCondition(ABC):
    history: HistoryType = field(default_factory=dict)
    stop_reason: str = field(default='', init=False)
    _generator: Optional[Generator[bool, None, None]] = field(default=None, init=False)

    @property
    def last_iteration(self) -> int:
        return max([max(i.keys()) for i in self.history.values()])

    @abstractmethod
    def stop_condition_generator(self) -> Generator[Tuple[bool, str], None, None]:
        """Create a generator that will yield stop condition results"""
        pass

    def _initialize_generator(self) -> None:
        """Initialize the generator"""
        if self._generator is None:
            self._generator = self.stop_condition_generator()

    def next(self, history: HistoryType) -> Tuple[bool, str]:
        """
        Executes the next step in the generation process, managing the state of the generator
        and responding to halt conditions. This method initializes the generator if necessary
        and updates the history attribute with the provided input. It then proceeds with the
        generator until a stop condition is explicitly met or the iteration ends.

        :param history: The current state or data that is fed into the generator. It guides
            the generator's iteration steps and contributes to determining when to stop.
        :type history: HistoryType
        :return: A tuple where the first element is a boolean indicating whether the
            generator should stop (True if it should stop), and the second element is
            a string providing the reason for stopping.
        :rtype: Tuple[bool, str]
        """
        while True:
            self._initialize_generator()
            try:
                self.history = history
                should_stop, reason = next(self._generator)
                return should_stop, reason
            except StopIteration:
                return True, f'Stop condition [{self.__class__.__name__}] met: {self.stop_reason}'



@dataclass
class Numerical(ABC):
    initial_state: Dict[str, Any] = field(init=False)
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

    def _check_stop_conditions(self) -> Generator[str, None, None]:
        """
        Checks if any of the specified stopping conditions are met and raises StopIteration
        if so.
        :yields: Always yields True until a stop condition is met
        """
        for iteration in range(self.max_iterations):
            self._iteration = iteration
            if iteration == 0:
                yield f'Initial Iteration completed'
                continue
            met_stop_conditions = []
            unmet_stop_conditions = []
            for stop_cond in self.stop_conditions:
                stop_cond_met, reason = stop_cond.next(self.history)
                condition_name = stop_cond.__class__.__name__
                if stop_cond_met:
                    met_stop_conditions.append(  f"Stop condition [{condition_name:<15}] met    : {reason}")
                else:
                    unmet_stop_conditions.append(f"Stop condition [{condition_name:<15}] not met: {reason}")
            status = '\n'.join(met_stop_conditions + unmet_stop_conditions)
            should_stop = len(unmet_stop_conditions) != len(self.stop_conditions)
            if should_stop:
                yield status
                break
            yield f'Iteration {iteration} completed\n{status}'

        else:
            # Handle max iterations reached
            logger.info(f"Stop condition max iterations reached ({self.max_iterations})")

    def run(self) -> HistoryType:
        """
        Run the numerical method using a for-loop with the generator method
        that checks stop conditions using StopIteration.
        :returns: Dict[str, Dict[int, Any]]: Complete history of the computation
        """
        self.history.clear()

        logger.info(f"Starting {self.__class__.__name__}")
        self.initialize()

        # Main iteration loop using the stop checker generator
        for status in self._check_stop_conditions():
            logger.info(f"{status}")
            logger.info(f"Iteration {self._iteration}....")
            state = self.step()
            self.record_state(state)
            logger.info(f"State: {json.dumps(state, indent=2)}")



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
        :param iteration: Iteration number
        :return: Dict[str, Any]: Variable values at specified iteration
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

    def export_to_latex(self, filepath: str, variables: List[str] = None,
                        iterations: List[int] = None, precision: int = 4,
                        caption: str = None, label: str = None) -> None:
        """
        Export history to a LaTeX table

        Args:
            filepath: Path to save the LaTeX table
            variables: List of variable names to include (None for all variables)
            iterations: List of iterations to include (None for all iterations)
            precision: Number of decimal places for floating point values
            caption: Optional caption for the table
            label: Optional label for the table
        """
        # Use all variables if none specified
        if variables is None:
            variables = list(self.history.keys())

        # Use all iterations if none specified
        if iterations is None:
            max_iter = self.last_iteration
            iterations = list(range(max_iter + 1))

        # Start building the LaTeX table
        latex_content = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\begin{{tabular}}{{c{'|c' * len(variables)}}}",
            "\\hline",
            f"Iteration & {' & '.join(variables)} \\\\ \\hline"
        ]

        # Add table rows
        for iter_num in iterations:
            try:
                state = self.get_history_at(iter_num)
                row_values = []
                for var in variables:
                    value = state.get(var)
                    if isinstance(value, (float, np.floating)):
                        formatted_val = f"{value:.{precision}f}"
                    else:
                        formatted_val = str(value)
                    row_values.append(formatted_val)
                latex_content.append(f"{iter_num} & {' & '.join(row_values)} \\\\")
            except ValueError:
                # Skip iterations that don't exist in history
                continue

        # Finish table
        latex_content.append("\\hline")
        latex_content.append("\\end{tabular}")

        # Add caption if provided
        if caption:
            latex_content.append(f"\\caption{{{caption}}}")

        # Add label if provided
        if label:
            latex_content.append(f"\\label{{{label}}}")

        latex_content.append("\\end{table}")

        # Write to file
        with open(filepath, 'w') as file:
            file.write('\n'.join(latex_content))

        logger.info(f"LaTeX table exported to {filepath}")
