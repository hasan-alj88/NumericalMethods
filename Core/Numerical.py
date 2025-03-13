import logging
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Generator, Set,Any, Dict

import pandas as pd

from Core.NumericalHistory import NumericalHistory
from StopConditions.StopConditionBase import StopCondition
from utils.log_config import get_logger


@dataclass
class Numerical(ABC):
    stop_conditions: List[StopCondition] = field(default_factory=list)
    max_iterations: int = 1_000
    _history: NumericalHistory = field(init=False, default=None)
    _iteration: int = field(default=0, init=False)
    console_log_level: int|str = field(default='OFF', init=False)
    _logger: logging.Logger = field(default=None, init=False)
    absolute_tolerance: float = field(default=1e-6)
    relative_tolerance:float = field(default=None)
    patience: int = field(default=3)
    t_label: str = field(default='t')
    y_label: str = field(default='y')

    @property
    def history(self) -> NumericalHistory:
        if self._history is None:
            self._history = NumericalHistory(console_log_level=self.console_log_level)
        return self._history

    @history.setter
    def history(self, value):
        self._history = value

    @property
    def logger(self):
        if self._logger is not None:
            return self._logger
        else:
            self._logger =get_logger(self.__class__.__name__, self.console_log_level)
        # if self.console_log_level == 'OFF':
        #     # remove console handler
        #     self._logger.handlers.pop()
        return self._logger

    # @logger.setter
    # def logger(self, value):
    #     self._logger = value

    @property
    def parameters(self) -> Set[str]:
        return set(self.initial_state.keys())

    @property
    @abstractmethod
    def initial_state(self) -> dict:
        pass

    def add_stop_condition(self, stop_condition: StopCondition) -> None:
        """Add a stop condition to the list of stop conditions"""
        self.stop_conditions.append(stop_condition)

    def initialize(self) -> None:
        self.history.data.clear()
        self._iteration = 0
        self.record_state(self.initial_state)
        self.logger.info(f"Initial state:{self.initial_state}")

    @abstractmethod
    def step(self) -> Dict[str, Any]:
        pass

    def record_state(self, state: dict) -> None:
        if set(state.keys()).difference(self.parameters).__len__() != 0:
            raise ValueError(f'Missing state parameters {set(state.keys()).difference(self.parameters)}'
                             f' in state :\n{state}\n'
                             f'Parameters: {self.parameters}\n'
                             f'State keys: {state.keys()}\n'
                             f'initial_state: {self.initial_state}')
        self.history.record_state(state)

    def _check_stop_conditions(self) -> Generator[str, None, None]:
        """
        Checks if any of the specified stopping conditions are met
        :yields: Always yields True until a stop condition is met
        """
        for iteration in range(1, self.max_iterations+1):
            self._iteration = iteration
            if iteration == 0:
                yield f'Initial Iteration completed'
                continue

            met_stop_conditions = []
            unmet_stop_conditions = []
            for stop_cond in self.stop_conditions:
                stop_cond_met, reason = stop_cond.next(self.history)
                condition_name = stop_cond.__str__()
                if stop_cond_met:
                    met_stop_conditions.append(  f"Stop condition [{condition_name:<20}] MET    : {reason}")
                else:
                    unmet_stop_conditions.append(f"Stop condition [{condition_name:<20}] NOT met: {reason}")

            status = '\n'.join(met_stop_conditions + unmet_stop_conditions)
            should_stop = len(unmet_stop_conditions) != len(self.stop_conditions)
            if should_stop:
                yield status
                break
            yield f'Iteration {iteration} completed\n{status}'

        else:
            self.logger.info(f"Stop condition max iterations reached ({self.max_iterations})")

    def run(self) -> pd.DataFrame:
        """
        Run the numerical method using a for-loop with the generator method
        that checks stop conditions using StopIteration.
        :returns: pd.DataFrame: Complete history of the computation
        """
        self.history = NumericalHistory()

        self.logger.info(f"Starting {self.__class__.__name__}")
        self.initialize()

        try:
            for status in self._check_stop_conditions():
                self.logger.info(f"{status}")
                self.logger.debug(f"Starting step {self.iteration}")
                assert len(self.history) > 0, 'history must be initialized before calling step()'
                state = self.step()
                self.record_state(state)
                self.logger.info(f"State: \n{state}\n")
        except Exception as e:
            self.logger.error(f"Exception occurred: {e}", exc_info=True)
            # self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise e
        finally:
            df = self.history.to_data_frame
            df.rename(columns={'t': self.t_label, 'y': self.y_label}, inplace=True)
            return df

    @property
    def iteration(self) -> int:
        """Current iteration number (read-only)"""
        return self._iteration

    @property
    def last_iteration(self) -> int:
        """Maximum number of iterations"""
        return len(self.history)
