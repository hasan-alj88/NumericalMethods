import struct
from typing import Callable, Tuple, List

import numpy as np

from Core import Numerical
from utils.ValidationTools import is_nan


class GAExtrema(Numerical):
    def __init__(
            self, *,
            num_generations: int = None,
            function: callable,
            find_max:bool = False,
            x_lower: float,
            x_upper: float,
            num_elites: int,
            num_parents_mating: int,
            mutation_probability: float,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.function: Callable = function
        self.find_max:bool = find_max
        self.x_lower: float = x_lower
        self.x_upper: float = x_upper
        self.max_iterations: int = num_generations or self.max_iterations
        self.num_elites: int = num_elites
        self.num_parents_mating: int = num_parents_mating
        self.mutation_probability: float = mutation_probability

    @property
    def num_genes(self) -> int:
        return self.num_elites + self.num_parents_mating

    @property
    def initial_state(self) -> dict:
        initial_genes: np.ndarray = np.random.uniform(self.x_lower, self.x_upper, self.num_genes)
        initial_genes: dict = {str(ind): val for ind, val in
                         sorted(enumerate(initial_genes), key=lambda x: self.function(x[1]), reverse=self.find_max)}
        initial_genes['f(x)'] = self.function(initial_genes['0'])
        return initial_genes

    def step(self) -> dict:
        self.logger.info(f'======= Generation {self.history.last_iteration} =====')
        previous_genes = self.history.last_state
        previous_genes_values = sorted(list(previous_genes.values()), key=self.function, reverse=self.find_max)
        next_genes = {}
        genes = np.arange(len(previous_genes_values)-1)

        # Elite Selection
        for ind, val in enumerate(previous_genes_values[:self.num_elites]):
            self.logger.info(f'Elite #{ind}:f({val}) = {self.function(val)}')
            next_genes[str(ind)] = val

        # Non-elite genes
        crossover_genes = genes[self.num_elites:].copy()
        np.random.shuffle(crossover_genes)

        for ind, (parent1_ind, parent2_ind) in enumerate(
                zip(crossover_genes, np.roll(crossover_genes, 1))
                ,start=self.num_elites):
            # Crossover
            self.logger.info(f'Crossover #{ind}')
            self.logger.info(f'Parent#{parent1_ind}: '
                             f'f({previous_genes[str(parent1_ind)]}) = '
                             f'{self.function(previous_genes[str(parent1_ind)])}')
            self.logger.info(f'Parent#{parent2_ind}: '
                             f'f({previous_genes[str(parent2_ind)]}) = '
                             f'{self.function(previous_genes[str(parent2_ind)])}')
            parent1 = previous_genes[str(parent1_ind)]
            parent2 = previous_genes[str(parent2_ind)]
            child1, child2 = self.crossover(parent1, parent2)
            child = np.random.choice([child1, child2])

            # Mutation
            if np.random.random() < self.mutation_probability:
                child = self.mutate(child)
                self.logger.info(f'Mutation applied to child: f({child}) = {self.function(child)}')

            next_genes[str(ind)] = child
            self.logger.info(f'Child -> f({child}) = {self.function(child)}')

        # Sort genes by fitness function
        next_genes = {ind: val for ind, val in
                      sorted(next_genes.items(), key=lambda _: self.function(_[1]), reverse=True)}
        next_genes['f(x)'] = self.function(next_genes['0'])
        self.logger.info(f'Best Gene: f({next_genes["0"]}) = {next_genes["f(x)"]}')
        return next_genes

    @staticmethod
    def float2bin(float_value: float) -> str:
        """Convert a float to a 32-bit binary string."""
        return format(struct.unpack('!I', struct.pack('!f', float_value))[0], '032b')

    def bin2float(self, bin_value: str) -> float:
        """Convert a 32-bit binary string back to a float."""
        bin_value = bin_value.zfill(32)  # Ensure the binary string is 32 bits
        float_value = struct.unpack('!f', struct.pack('!I', int(bin_value, 2)))[0]

        if is_nan(float_value) or np.isinf(float_value):
            # Not a valid IEEE binary representation
            # select random float within range
            self.logger.warning(f'Invalid binary representation: {bin_value}. ')
            float_value = np.random.uniform(self.x_lower, self.x_upper)
            self.logger.warning(f'Using random value: {float_value}')
        return float_value

    def crossover(self, parent1: float, parent2: float) -> Tuple[float, float]:
        # No Incest
        for _ in range(1000):
            if parent1 != parent2:
                break
            parent1 = self.mutate(parent1)
            parent2 = self.mutate(parent2)

        # Convert to binary string
        parent1_bin = self.float2bin(parent1)
        parent2_bin = self.float2bin(parent2)

        # 32-bit
        m = np.random.randint(1, 32)

        # apply cross-over
        child1_bin = parent1_bin[:m] + parent2_bin[m:]
        child2_bin = parent2_bin[:m] + parent1_bin[m:]

        # convert back to float
        child1 = self.bin2float(child1_bin)
        child2 = self.bin2float(child2_bin)

        # Ensure children are within bounds
        child1 = self.clamp_to_range(child1)
        child2 = self.clamp_to_range(child2)

        return child1, child2

    def mutate(self, chromosome: float) -> float:
        chromosome_bin: List[str] = list(self.float2bin(chromosome))
        flip_bin_index = np.random.randint(len(chromosome_bin))
        flip_bit = chromosome_bin[flip_bin_index]
        chromosome_bin[flip_bin_index] = '0' if flip_bit == '1' else '1'
        chromosome_bin: str = ''.join(chromosome_bin)
        chromosome = self.bin2float(chromosome_bin)

        # Ensure mutated value is within bounds
        chromosome = self.clamp_to_range(chromosome)

        return chromosome

    def clamp_to_range(self, value: float) -> float:
        """Ensures the value stays within the specified range."""
        if self.x_lower <= value <= self.x_upper:
            return value
        self.logger.warning(f'Value {value} outside of range [{self.x_lower}, {self.x_upper}]. ')
        float_value = np.random.uniform(self.x_lower, self.x_upper)
        self.logger.warning(f'Using random value: {float_value}')
        return float_value

    def best_solution(self):
        solutions = self.history.to_data_frame
        solutions = solutions.sort_values(by='f(x)', ascending=self.find_max)
        return solutions.iloc[0, 0], self.function(solutions.iloc[0, 0])


if __name__ == "__main__":
    ga = GAExtrema(
        function=lambda x: (x - 0.75) ** 2,
        x_lower=0,
        x_upper=1,
        num_generations=30,
        num_elites=10,
        num_parents_mating=5,
        mutation_probability=0.1,
    )
    print(ga.run())