from itertools import product
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import pandas as pd


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

    # a_i = f_i
    a = y_data[:n_intervals]

    unknowns = [f'{v}_{i}' for v, i in product(['b','c','d'],range(n_intervals))]
    equations = pd.DataFrame(columns=unknowns+['LHS'])
    # f_{i+1}-f_i = b_i h_i + c_i h_i^2 + d_i h_i^3
    for i in range(n_intervals):
        equations.loc[i, f'b_{i}']=h[i]
        equations.loc[i, f'c_{i}']=h[i]**2
        equations.loc[i, f'd_{i}']=h[i]**3
        equations.loc[i, 'LHS']=delta_y[i]

    # 0 = b_i - b_{i+1} + 2h_i c_i + 3h_i^2 d_i
    k = len(equations)
    for i in range(n_intervals-1):
        equations.loc[k+i, f'b_{i}']=1
        equations.loc[k+i, f'b_{i+1}']=-1
        equations.loc[k+i, f'c_{i}']=2*h[i]
        equations.loc[k+i, f'd_{i}']=3*h[i]**2
        # LHS = 0 implied

    # 0 = c_i - c_{i+1} + 3h_i d_i
    k = len(equations)
    for i in range(n_intervals-1):
        equations.loc[k+i+1, f'c_{i}']=1
        equations.loc[k+i+1, f'c_{i+1}']=-1
        equations.loc[k+i+1, f'd_{i}']=3*h[i]
        # LHS = 0 implied

    k = len(equations)
    if end_condition.lower() == "natural":
        # c_0 = 0
        equations.loc[k+1, 'c_0']=1
        # LHS = 0 implied

        # c_n = 0
        equations.loc[k+2, f'c_{n_intervals-1}']=1
        # LHS = 0 implied

    elif end_condition.lower() == "not-a-knot":
        # d_0 - d_1 = 0
        equations.loc[k+1, 'd_0']=-1
        equations.loc[k+1, 'd_1']=1
        # LHS = 0 implied

        # d_{n-1} - d_n = 0
        equations.loc[k + 2, f'd_{n_intervals-2}'] = -1
        equations.loc[k + 2, f'd_{n_intervals-1}'] = 1
        # LHS = 0 implied

    else:
        raise ValueError(f"end_condition must be 'natural' or 'not-a-knot' Got {end_condition} which is not available")

    equations.fillna(0, inplace=True)
    m = equations.loc[:,unknowns]
    lhs = equations.loc[:,'LHS']
    lhs = lhs.reindex(m.index)    # ensure the equation correctly ordered
    solution = pd.Series(np.linalg.solve(m.values, lhs.values), index=m.columns)

    # extract solution
    b = [solution[f'b_{i}'] for i in range(n_intervals)]
    c = [solution[f'c_{i}'] for i in range(n_intervals)]
    d = [solution[f'd_{i}'] for i in range(n_intervals)]
    lower_bound = x_data[:-1]
    upper_bound = x_data[1:]

    spline_functions = []
    for ai,bi,ci,di,x_min,x_max in zip(a,b,c,d,lower_bound,upper_bound):
        spline_functions.append({
            's': ai+bi*(x-x_min)+ci*(x-x_min)**2+di*(x-x_min)**3,
            'interval':(x_min,x_max),
        })

    return spline_functions


def spline_to_latex(spline_fns: List[dict]):
    latex_str = r"\begin{equation}"+'\n'
    latex_str += r"f(x) = "+'\n'
    latex_str += r"\begin{cases}"+'\n'
    for i,fn in enumerate(spline_fns):
        x_min, x_max = sorted(list(fn.get('interval')))
        s_latex = sp.latex(fn.get('s'))
        latex_str += fr"s_{i}(x)={s_latex} & {x_min:0.3g} \leq x \leq {x_max:0.3g} \\"+'\n'
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
    for fn in spline:
        x_min, x_max = sorted(list(fn.get('interval')))
        s = fn.get('s')
        assert s is not None
        f = sp.lambdify(x_sym, s, "numpy")

        x_points = np.linspace(x_min, x_max, 1000)
        y_points = f(x_points)

        ax.plot(x_points, y_points, **kwargs)

    return ax