from decimal import Decimal

from pydantic.v1 import BaseModel


class BracketingState(BaseModel):
    a: Decimal
    b: Decimal
    fa: Decimal
    fb: Decimal
    root: Decimal
    log: str

    @property
    def bracket_size(self)-> Decimal:
        return abs(self.b - self.a)