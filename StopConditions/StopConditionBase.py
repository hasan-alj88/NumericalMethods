from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Generator, Tuple, Optional, Dict

import pandas as pd

from Core.NumericalHistory import NumericalHistory
from utils.log_config import get_logger


@dataclass
class StopCondition(ABC):
    history: NumericalHistory = field(default_factory=NumericalHistory)
    stop_condition_history: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)
    stop_reason: str = field(default='', init=False)
    _generator: Optional[Generator[Tuple[bool, str], None, None]] = field(default=None, init=False)

    def __post_init__(self):
        self.logger = get_logger(self.__class__.__name__)

    @property
    def last_iteration(self) -> int:
        return len(self.history)

    @abstractmethod
    def stop_condition_generator(self) -> Generator[Tuple[bool, str], None, None]:
        """Create a generator that will yield stop condition results"""
        pass

    def _initialize_generator(self) -> None:
        """Initialize the generator"""
        if self._generator is None:
            self._generator = self.stop_condition_generator()

    def update_stop_history(self, current_condition_params: Dict[str, any]) -> None:
        """Update the stop history with the current condition parameters"""
        new_row = pd.DataFrame([current_condition_params])
        self.stop_condition_history = pd.concat([self.stop_condition_history, new_row], ignore_index=True)

    def next(self, history: NumericalHistory) -> Tuple[bool, str]:
        """
        Executes the next step in the generation process, managing the state of the generator
        and responding to halt conditions.

        :param history: The current state NumericalHistory.
        :type history: NumericalHistory
        :return: A tuple where the first element is a boolean indicating whether the
            generator should stop, and the second element is the reason for stopping.
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
