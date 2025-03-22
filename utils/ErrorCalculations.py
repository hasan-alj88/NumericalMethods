from decimal import Decimal
from typing import Dict, Any

from utils.ValidationTools import is_nan


def absolute_error(x_estimate, x_exact):
    return abs(x_estimate - x_exact)


def relative_error(x_estimate, x_exact):
    return abs(x_estimate - x_exact) / abs(x_exact)



def tolerance_condition(
        first_value: Decimal,
        second_value: Decimal,
        absolute_tolerance: Decimal=None,
        relative_tolerance: Decimal=None
) -> Dict[str, Any]:
    if is_nan(first_value) or is_nan(second_value):
        raise ValueError('Cannot calculate tolerance condition for NaN values')

    # Calculate differences
    abs_diff = abs(Decimal(second_value) - Decimal(first_value))
    rel_diff = abs_diff / abs(Decimal(first_value)) if first_value != 0 else Decimal('inf')

    # Determine if we've reached a plateau based on tolerances
    abs_tol_met = absolute_tolerance is not None and abs_diff <= absolute_tolerance
    rel_tol_met = relative_tolerance is not None and rel_diff <= relative_tolerance

    # Stop condition logic: if either tolerance is met (or both)
    stop_condition = abs_tol_met or rel_tol_met

    return dict(
        abs_diff=abs_diff,
        rel_diff=rel_diff,
        abs_tol_met=abs_tol_met,
        rel_tol_met=rel_tol_met,
        stop_condition=stop_condition
    )