def absolute_error(x_estimate, x_exact):
    return abs(x_estimate - x_exact)


def relative_error(x_estimate, x_exact):
    return abs(x_estimate - x_exact) / abs(x_exact)