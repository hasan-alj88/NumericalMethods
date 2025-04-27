from typing import List

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def cubic_spline_functions(
        x_data: np.ndarray,
        y_data: np.ndarray,
        end_condition: str = "natural",
        x: sp.Symbol = sp.Symbol("x")
) -> List[dict]:
    n = len(x_data)
    n_intervals = n - 1

    h = np.diff(x_data)  # h[i] = x[i+1] - x[i]
    delta_y = np.diff(y_data)  # delta_y[i] = y[i+1] - y[i]

    ### [b_i] h_i + [c_i] h_i^2 + [d_i] h_i^3 & = f_{i+1} - f_i ###
    mb1 = np.diag(h)  # [b_i] h_i
    mc1 = np.diag(h ** 2)  # + [c_i] h_i^2
    md1 = np.diag(h ** 3)  # + [d_i] h_i^3
    l1 = delta_y.copy()  # = f_{i+1} - f_i

    ### - [b_i] + [b_{i+1}] - 2[c_i] h_i - 3[d_i] h_i^2 & = 0 ###
    mb2 = np.zeros((n_intervals - 1, n_intervals))
    for i in range(n_intervals - 1):
        mb2[i, i] = -1  # - [b_i]
        mb2[i, i + 1] = 1  # + [b_{i+1}]

    # Fix dimension issues by using zeros matrices with proper dimensions
    mc2 = np.zeros((n_intervals - 1, n_intervals))
    md2 = np.zeros((n_intervals - 1, n_intervals))

    # Fill in diagonal elements
    for i in range(n_intervals - 1):
        mc2[i, i] = -2 * h[i]  # - [c_i]2 h_i
        md2[i, i] = -3 * h[i] ** 2  # - [d_i]3 h_i^2

    l2 = np.zeros(n_intervals - 1)  # = 0

    ### - [c_i] + [c_{i+1}]   - 3[d_i] h_i & = 0  ###
    mb3 = np.zeros((n_intervals - 1, n_intervals))  # 0
    mc3 = np.zeros((n_intervals - 1, n_intervals))  # 0
    md3 = np.zeros((n_intervals - 1, n_intervals))  # Initialize with proper dimensions

    for i in range(n_intervals - 1):
        mc3[i, i] = -1  # - [c_i]
        mc3[i, i + 1] = 1  # + [c_{i+1}]
        md3[i, i] = -3 * h[i]  # - [d_i]3 h_i

    l3 = np.zeros(n_intervals - 1)  # = 0

    if end_condition == "natural":
        ### c_0 = 0, c_{n-2} = 0 ###
        mb4 = np.zeros((2, n_intervals))
        mc4 = np.zeros((2, n_intervals))
        mc4[0, 0] = 1  # c0
        mc4[1, -1] = 1  # c_{n-2}
        md4 = np.zeros((2, n_intervals))
        l4 = np.zeros((2,))

    elif end_condition == "not-a-knot":
        ### d_0 = d_1, d_{n-3} = d_{n-2} ###
        mb4 = np.zeros((2, n_intervals))
        mc4 = np.zeros((2, n_intervals))
        md4 = np.zeros((2, n_intervals))
        md4[0, 0] = 1  # d_0
        md4[0, 1] = -1  # -d_1
        md4[1, -2] = -1  # -d_{n-2}
        md4[1, -1] = 1  # d_{n-3} (fixed index)
        l4 = np.zeros((2,))  # =0
    else:
        raise ValueError("end_condition must be 'natural' or 'not-a-knot'")

    # construct the M matrix
    m1 = np.concatenate((mb1, mc1, md1), axis=1)
    m2 = np.concatenate((mb2, mc2, md2), axis=1)
    m3 = np.concatenate((mb3, mc3, md3), axis=1)
    m4 = np.concatenate((mb4, mc4, md4), axis=1)

    assert m1.shape[1] == m2.shape[1] == m3.shape[1] == m4.shape[1], "Column dimensions must match for concatenation"

    m = np.concatenate((m1, m2, m3, m4), axis=0)  # Combine all rows

    # construct L matrix
    l = np.concatenate((l1, l2, l3, l4), axis=0)

    z = np.linalg.solve(m, l)

    # Extract coefficients
    b = z[:n_intervals]
    c = z[n_intervals:2 * n_intervals]
    d = z[2 * n_intervals:]
    a = y_data[:-1]  # a coefficients are the y values at the left endpoint of each interval

    assert b.shape == c.shape == d.shape

    spline_functions = []
    for i, (aa, bb, cc, dd) in enumerate(zip(a, b, c, d)):
        x_i = x_data[i]

        # Spline function
        t = x - x_i
        spline = aa + bb * t + cc * t ** 2 + dd * t ** 3

        spline_functions.append({
            's': spline,
            'interval': (x_data[i], x_data[i + 1])
        })

    return spline_functions


def spline_to_latex(spline_fns: List[dict]):
    latex_str = r"\begin{equation}"+'\n'
    latex_str += r"f(x) = "+'\n'
    latex_str += r"\begin{cases}"+'\n'
    for fn in spline_fns:
        x_min, x_max = sorted(list(fn.get('interval')))
        s_latex = sp.latex(fn.get('s').simplify())
        latex_str += fr"{s_latex} & {x_min:0.3g} \leq x \leq {x_max:0.3g} \\"+'\n'
    latex_str += r"\end{cases}"+'\n'
    latex_str += r"\end{equation}"
    return latex_str

def evaluate_spline(spline_fns: List[dict], xx:np.ndarray)->np.ndarray:
    yy = np.zeros_like(xx)
    for i, xi in enumerate(xx):
        for spline in spline_fns:
            x_min, x_max = sorted(list(spline.get('interval')))
            if x_min < xi < x_max:
                s = spline.get('s')
                v = list(s.free_symbols)[0]
                yy[i] = s.subs({v:xi})
                break
        else:
            yy[i] = np.nan
    assert len(yy) == len(xx), "len(yy) must equal len(xx)"
    return np.array(yy)




def plot_splines(spline: List[dict], ax:plt.Axes=None, **kwargs)->plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    x_sym = list(spline[0]['s'].free_symbols)[0]  # Get the symbol used in expressions

    # Create a more direct plotting approach using numpy instead of sympy.plotting
    for fn in spline:
        x_min, x_max = sorted(list(fn.get('interval')))
        s = fn.get('s')
        assert s is not None

        # Convert sympy expression to numpy function
        f = sp.lambdify(x_sym, s, "numpy")

        # Create x points for smooth plotting
        x_points = np.linspace(x_min, x_max, 1000)
        y_points = f(x_points)

        # Plot the segment
        ax.plot(x_points, y_points, **kwargs)

    return ax