from matplotlib import pyplot as plt


def draw_line(point1, point2, ax=None, *args, **kwargs):
    """
    Draw a straight line between two points.

    Parameters:
    -----------
    point1 : tuple or list
        The first point as (x, y)
    point2 : tuple or list
        The second point as (x, y)
    ax : matplotlib.axes.Axes, optional
        The axis to draw on. If None, the current axis will be used.

    Returns:
    --------
    matplotlib.axes.Axes
        The axis with the line drawn on it
    """
    # Get the x and y coordinates from the points
    x1, y1 = point1
    x2, y2 = point2

    # If no axis is provided, get the current axis
    if ax is None:
        ax = plt.gca()

    # Draw the line
    ax.plot([x1, x2], [y1, y2], *args, **kwargs)

    return ax