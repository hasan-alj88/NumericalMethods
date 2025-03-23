from functools import partial
from typing import Dict, Any

import numpy as np

from FindRoots.BracketingMethods.BracketingMethods import BracketingMethods


class FalsePositionMethod(BracketingMethods):
    @property
    def initial_state(self) -> dict:
        a, b = self.a, self.b
        fa, fb = self.function(a), self.function(b)
        c = a - fb * (a - b) / (fa - fb)
        return dict(
            a=a,
            b=b,
            c=c,
            fa=fa,
            fb=fb,
            fc=self.function(c),
            log='Initial state'
        )

    def step(self) -> Dict[str, Any]:
        a = self.history['a']
        b = self.history['c']

        fa, fb = self.function(a), self.function(b)

        c = a - fb * (a - b) / (fa - fb)
        fc = self.function(c)

        self.logger.info(f'f({c:0.3e}) = {fc:0.3e}')

        tol_kwargs= dict(b=0.0)
        if self.absolute_tolerance is not None:
            tol_kwargs['atol'] = self.absolute_tolerance
        if self.relative_tolerance is not None:
            tol_kwargs['rtol'] = self.relative_tolerance
        is_close = partial(np.isclose, **tol_kwargs)

        if fa * fc < 0:
            a, b = a, c
            log = 'Root in [a,c]'
        elif fb * fc < 0:
            a, b = c, b
            log = 'Root in [c,b]'
        elif is_close(fa):
            a, b = a, a
            log = 'Root found at a'
        elif is_close(fb):
            a, b = b, b
            log = 'Root found at b'
        elif is_close(fc):
            a, b = c, c
            log = 'Root found at c'
        else:
            raise ValueError("False position method failed to converge")

        return dict(
            a=a,
            b=b,
            c=c,
            fa=fa,
            fb=fb,
            fc=fc,
            log=log
        )



