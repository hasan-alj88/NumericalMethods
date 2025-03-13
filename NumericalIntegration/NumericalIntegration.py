import numpy as np
from utils.ValidationTools import function_arg_count


def trapezoidal_integration(
        function: callable,
        a: float = 0.0,
        b: float = 1.0,
        h: float = 0.1,
) -> float:
    """
    I = \int_{x_0}^{x_n} f(x) dx = \frac{h}{2} [f(x_0) + 2 \sum^{n-1}_{i=1} f(x_i) + f(x_n)]
    :param function: f(x)
    :param a: x_0
    :param b: x_n
    :param h: step
    :return: \int_{x_0}^{x_n} f(x) dx
    """
    if function_arg_count(function) != 1:
        raise ValueError(f'Function must have exactly one argument not {function_arg_count(function)}')

    xi = np.arange(a, b, h)[1:-1]
    return h / 2 * (function(a) + 2 * np.sum(function(xi)) + function(b))

def simpson_integration(
        function: callable,
        a: float = 0.0,
        b: float = 1.0,
        h: float = 0.1,
) -> float:
    """
    I = \int_{x_0}^{x_n} f(x) dx = \frac{b-a}{3n} [f(a) +
                                    4 \sum^{n-1}_{i=Odd} f(x_i) +
                                    2 \sum^{n-1}_{i=Even} f(x_i)+ f(b)]
    :param function: f(x)
    :param a: x_0
    :param b: x_n
    :param h: step
    :return: I = \int_{x_0}^{x_n} f(x) dx
    """
    if function_arg_count(function) != 1:
        raise ValueError(f'Function must have exactly one argument not {function_arg_count(function)}')

    xi = np.arange(a, b, h)[1:-1]
    xi_odd = xi[1::2]
    xi_even = xi[::2]
    n = len(xi)
    return (b - a) / (3 * n) * (
            function(a) +
            4 * np.sum(function(xi_odd)) +
            2 * np.sum(function(xi_even)) +
            function(b))

def simpsons_3_8_integration(
        function: callable,
        a: float = 0.0,
        b: float = 1.0,
        h: float = 0.1,
) -> float:
    """
    I = \int_{x_0}^{x_n} f(x) dx = \frac{3h}{8}[f(a)+ 3 \sum^{n-1}_{i=1} f(x_i) + f(b)]
    :param function: f(x)
    :param a: x_0
    :param b: x_n
    :param h: step
    :return: I = \int_{x_0}^{x_n} f(x) dx
    """
    if function_arg_count(function) != 1:
        raise ValueError(f'Function must have exactly one argument not {function_arg_count(function)}')

    xi = np.arange(a, b, h)[1:-1]
    return (3 * h / 8) * (function(a) + 3 * np.sum(function(xi)) + function(b))


def simpsons_1_3_integration(
        function: callable,
        a: float = 0.0,
        b: float = 1.0,
        h: float = 0.1,
) -> float:
    """
    I = \int_{x_0}^{x_n} f(x) dx = \frac{h}{3}[f(x_0) + 4f(x_1) + 2f(x_2) + 4f(x_3) + ... + 2f(x_{n-2}) + 4f(x_{n-1}) + f(x_n)]

    This is the standard Simpson's 1/3 rule for numerical integration.

    :param function: f(x)
    :param a: x_0
    :param b: x_n
    :param h: step
    :return: I = \int_{x_0}^{x_n} f(x) dx
    """
    if function_arg_count(function) != 1:
        raise ValueError(f'Function must have exactly one argument not {function_arg_count(function)}')

    # Generate all points including endpoints
    x = np.arange(a, b + h / 2, h)  # Adding h/2 to ensure b is included
    n = len(x) - 1  # Number of intervals

    # Check if number of intervals is even (required for Simpson's 1/3)
    if n % 2 != 0:
        raise ValueError(f"Simpson's 1/3 rule requires an even number of intervals, got {n}")

    # Compute weights for all interior points
    coeffs = np.ones(n + 1)
    coeffs[1:n:2] = 4  # Odd indices (except endpoints)
    coeffs[2:n:2] = 2  # Even indices (except endpoints)

    # Compute the integral
    return (h / 3) * np.sum(coeffs * function(x))


def booles_rule_integration(
        function: callable,
        a: float = 0.0,
        b: float = 1.0,
        h: float = 0.1,
) -> float:
    """
    I = \int_{x_0}^{x_4} f(x) dx = \frac{2h}{45}[7f(x_0) + 32f(x_1) + 12f(x_2) + 32f(x_3) + 7f(x_4)]

    Boole's rule is a 5-point Newton-Cotes formula for numerical integration.

    :param function: f(x)
    :param a: x_0
    :param b: x_n
    :param h: step
    :return: I = \int_{x_0}^{x_n} f(x) dx
    """
    if function_arg_count(function) != 1:
        raise ValueError(f'Function must have exactly one argument not {function_arg_count(function)}')

    # For Boole's rule, we need to ensure we have segments of 4 intervals each
    x = np.arange(a, b + h / 2, h)
    n = len(x) - 1  # Number of intervals

    if n % 4 != 0:
        raise ValueError(f"Boole's rule requires the number of intervals to be a multiple of 4, got {n}")

    # Calculate number of segments
    num_segments = n // 4
    result = 0.0

    # Apply Boole's rule to each segment
    for i in range(num_segments):
        start_idx = i * 4
        segment_points = x[start_idx:start_idx + 5]  # 5 points for each segment
        segment_values = function(segment_points)

        # Apply Boole's rule weights
        segment_result = (2 * h / 45) * (7 * segment_values[0] + 32 * segment_values[1] +
                                         12 * segment_values[2] + 32 * segment_values[3] +
                                         7 * segment_values[4])
        result += segment_result

    return result


def newton_cotes_6_point_integration(
        function: callable,
        a: float = 0.0,
        b: float = 1.0,
        h: float = 0.1,
) -> float:
    """
    I = \int_{x_0}^{x_5} f(x) dx = \frac{5h}{288}[19f(x_0) + 75f(x_1) + 50f(x_2) + 50f(x_3) + 75f(x_4) + 19f(x_5)]

    6-point Newton-Cotes formula for numerical integration.

    :param function: f(x)
    :param a: x_0
    :param b: x_n
    :param h: step
    :return: I = \int_{x_0}^{x_n} f(x) dx
    """
    if function_arg_count(function) != 1:
        raise ValueError(f'Function must have exactly one argument not {function_arg_count(function)}')

    # For 6-point Newton-Cotes, we need segments of 5 intervals each
    x = np.arange(a, b + h / 2, h)
    n = len(x) - 1  # Number of intervals

    if n % 5 != 0:
        raise ValueError(
            f"6-point Newton-Cotes formula requires the number of intervals to be a multiple of 5, got {n}")

    # Calculate number of segments
    num_segments = n // 5
    result = 0.0

    # Apply 6-point Newton-Cotes to each segment
    for i in range(num_segments):
        start_idx = i * 5
        segment_points = x[start_idx:start_idx + 6]  # 6 points for each segment
        segment_values = function(segment_points)

        # Apply 6-point Newton-Cotes weights
        segment_result = (5 * h / 288) * (19 * segment_values[0] + 75 * segment_values[1] +
                                          50 * segment_values[2] + 50 * segment_values[3] +
                                          75 * segment_values[4] + 19 * segment_values[5])
        result += segment_result

    return result
