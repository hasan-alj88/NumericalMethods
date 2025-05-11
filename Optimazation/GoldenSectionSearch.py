from jinja2.filters import do_xmlattr

from Core import Numerical
from StopConditions.StopIfEqual import StopIfZero


class GoldenSectionSearch(Numerical):
    def __init__(
            self,*,
            function: callable,
            x_lower: float,
            x_upper: float,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.function: callable = function
        self.x_lower: float = x_lower
        self.x_upper: float = x_upper

        self.add_stop_condition(StopIfZero(tracking='dx', patience=self.patience,
                                           absolute_tolerance=self.absolute_tolerance,
                                           relative_tolerance=self.relative_tolerance)
        )

    @property
    def initial_state(self) -> dict:
        return dict(
            xl=self.x_lower,
            xu=self.x_upper,
            fl=self.function(self.x_lower),
            fu=self.function(self.x_upper),
            dx=abs(self.x_upper - self.x_lower),
            log='initial state'
        )

    def step(self) -> dict:
        previous_state = self.history.last_state
        xl = previous_state['xl']
        xu = previous_state['xu']
        fl = previous_state['fl']
        fu = previous_state['fu']

        phi = (1 + 5 ** 0.5) / 2
        r = 1 / phi
        dx = abs(xu - xl)
        x1 = xu - r * dx
        x2 = xl + r * dx

        f1 = self.function(x1)
        f2 = self.function(x2)

        if f1 < f2:
            xl, xu = xl, x2
            fl, fu = fu, f2
            log = f'f1 < f2, x2 is better'
        elif f1 > f2:
            xl, xu = x1, xu
            fl, fu = f1, fu
            log = f'f1 > f2, x1 is better'
        else:
            xl, xu = x1, x1
            fl, fu = f1, f1
            log = f'f1 == f2, x1 is better'
        return dict(
            xl=xl,
            xu=xu,
            fl=fl,
            fu=fu,
            dx=dx,
            log=log
        )

if __name__ == "__main__":
    xeb = GoldenSectionSearch(function=lambda x: (x-0.75)**2, x_lower=0, x_upper=1, max_iterations=30)
    result = xeb.run()
    print(result)