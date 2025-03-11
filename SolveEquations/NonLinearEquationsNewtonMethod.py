from dataclasses import dataclass, field
from typing import List

import numpy as np
import sympy

from Core import Numerical
from StopConditions.StopAtOrderOfMagnitudeIncrease import StopAtOrderOfMagnitudeIncrease
from StopConditions.StopIfEqual import StopIfZero
from StopConditions.StopIfNaN import StopIfNaN
from utils.ValidationTools import is_nan


@dataclass
class NonLinearEquationsNewtonMethod(Numerical):
    equations:List[sympy.Expr] = field(default_factory=list)
    jacobian:sympy.Matrix = field(default=None, init=False)
    initial_guess:dict = field(default_factory=dict)
    variables:List[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        if len(self.equations) == 0:
            raise ValueError("Must specify at least one equation")

        variables = set()
        for eq in self.equations:
            variables |= {str(v) for v in eq.free_symbols}
        self.variables: list = sorted(variables)

        self.jacobian = sympy.Matrix(self.equations).jacobian(list(self.variables))

        missing_initial_guess = set(self.variables) - set(self.initial_guess.keys())
        if missing_initial_guess:
            raise ValueError(f"Missing initial guess for {missing_initial_guess}")

        self.add_stop_condition(StopIfZero(tracking='residue', patience=self.patience,
                                           absolute_tolerance=self.absolute_tolerance,
                                           relative_tolerance=self.relative_tolerance))
        self.add_stop_condition(StopAtOrderOfMagnitudeIncrease(
            tracking='residue', patience=self.patience,
        ))
        self.add_stop_condition(StopIfNaN(track_variables=['residue'] + list(self.variables)))

    def get_values(self, iteration: int)->dict:
        return {k:v for k,v in self.history.data[iteration].items() if k in list(map(str, self.variables))}

    @property
    def initial_state(self) -> dict:
        f0 = {f'f_{i}': f.subs(self.initial_guess).evalf() for i, f in enumerate(self.equations)}
        f0_values = list(f0.values())
        residue = sympy.Matrix(f0_values).norm(ord=2)
        return {
            **self.initial_guess,
            **f0,
            'residue': residue
        }

    def _no_solution(self)->dict:
        x_dict = {v: np.nan for v in self.variables}
        return {
            **x_dict,
            **{f'f_{i}': eq.subs(x_dict).evalf() for i, eq in enumerate(self.equations)},
            'residue': np.nan
        }

    def step(self):
        """
        $$ X_{k+1} = X_{k} - J(X_k)^{-1} F(X_k) $$
        """
        # Get current variable values
        xk_sub = self.get_values(-1)
        #
        # if np.any([is_nan(_) for _ in self.history.data[-1].values()]):
        #     return self._no_solution()

        xk_values = [xk_sub[v] for v in self.variables]


        xk = sympy.Matrix(xk_values)
        fk = sympy.Matrix([eq.subs(xk_sub) for eq in self.equations])
        jk = self.jacobian.subs(xk_sub)

        try:
            # Calculate ΔX_k = -J(X_k)^(-1) · F(X_k)
            delta_x = -jk.inv() @ fk

            # Calculate X_{k+1} = X_k + ΔX_k
            xk1 = xk + delta_x
        except sympy.matrices.exceptions.NonInvertibleMatrixError:
            self.logger.warning(f"Jacobian not invertible: {jk}")
            return self._no_solution()

        # Convert result to dictionary
        xk1_dict = {v: xk1[i].evalf()  if hasattr(xk[i], 'evalf') else xk1[i]
                    for i, v in enumerate(self.variables)}
        fk1 = {f'f_{i}': eq.subs(xk1_dict).evalf() for i, eq in enumerate(self.equations)}

        return {
            **xk1_dict,
            **fk1,
            'residue': sympy.Matrix([f for f in fk1.values()]).norm(ord=2).evalf()
        }



