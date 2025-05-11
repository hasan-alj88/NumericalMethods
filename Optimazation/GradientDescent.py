from typing import Dict, List

import numpy as np
import sympy

from Core import Numerical
from StopConditions.StopIfEqual import StopIfEqual


class GradientDescent(Numerical):
    def __init__(
            self, *,
            function: sympy.Expr,
            learning_rate: float,
            initial_guess: dict = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.function: sympy.Expr = function
        self.learning_rate: float = learning_rate

        # Validate initial_guess
        assert initial_guess is not None, f'initial_guess cannot be None'
        assert isinstance(initial_guess, dict), f'initial_guess must be a dict'
        assert set(initial_guess.keys()) == set(self.variables), (
            f'initial_guess must contain all variables.'
            f'Missing: {set(self.variables) - set(initial_guess.keys())}')
        self._initial_guess = initial_guess

        self.gradient_expressions = {
            str(xi):self.function.diff(xi)
            for xi in self.function.free_symbols
        }

        self.add_stop_condition(StopIfEqual(tracking='GradientNorm', patience=self.patience,
                                           absolute_tolerance=self.absolute_tolerance,
                                           relative_tolerance=self.relative_tolerance))

    @property
    def variables(self)->List[str]:
        return [str(xi) for xi in self.function.free_symbols]

    def gradient(self, values: dict)->Dict[str, float]:
        grad_dict = {}
        grad_values = []
        for xi, grad_expr in self.gradient_expressions.items():
            grad_values.append(np.float64(grad_expr.subs(values).evalf()))
            grad_dict[f'Gradient_{xi}'] = grad_values[-1]
        if len(grad_values) == 1:
            grad_dict['GradientNorm'] = grad_values[0]
        else:
            grad_dict['GradientNorm'] = np.linalg.norm(grad_values)
        return grad_dict

    @property
    def initial_state(self) -> dict:
        initial_state_dict = self._initial_guess.copy()
        initial_state_dict['f(x)'] = self.function.subs(initial_state_dict).evalf()
        grad = self.gradient(initial_state_dict)
        initial_state_dict.update(grad)
        return initial_state_dict

    def step(self) -> dict:
        previous_state = self.history.last_state
        next_state = previous_state.copy()
        previous_state_sub = {k:v for k,v in previous_state.items() if k in self.variables}

        current_grad = self.gradient(previous_state_sub)
        for gxi, grad_xi in current_grad.items():
            if gxi == 'GradientNorm':
                continue
            xi = gxi.split('Gradient_')[-1]
            next_state[xi] = previous_state[xi] - self.learning_rate * grad_xi

        next_state['f(x)'] = self.function.subs(next_state).evalf()
        next_state.update(current_grad)
        next_state['GradientNorm'] = self.gradient(next_state)['GradientNorm']
        return next_state


if __name__ == "__main__":
    import sympy

    # Example 1: Simple quadratic function
    x = sympy.Symbol('x')
    f = x ** 2 + 2 * x + 1  # (x+1)^2, minimum at x=-1

    print("Testing with f(x) = x^2 + 2x + 1")
    gd = GradientDescent(function=f, learning_rate=0.1, initial_guess={'x': 0}, max_iterations=20)
    result = gd.run()
    print(result)

    # Example 2: Multivariable function
    x, y = sympy.symbols('x y')
    f2 = x ** 2 + 2 * y ** 2 + x * y  # Minimum at (0,0)

    print("\nTesting with f(x,y) = x^2 + 2y^2 + xy")
    gd2 = GradientDescent(
        function=f2,
        learning_rate=0.1,
        initial_guess={'x': 1, 'y': 1},
        max_iterations=20)
    result2 = gd2.run()
    print(result2)