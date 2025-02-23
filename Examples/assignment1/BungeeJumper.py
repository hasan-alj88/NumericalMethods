from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Numerical import df_to_latex
from ODE.ExplicitFirstOrderODE.FirstOrderAutonomousODEEulerMethod import FirstOrderAutonomousODEEulerMethod
from ODE.ExplicitFirstOrderODE.FirstOrderAutonomousODEHeunMethod import FirstOrderAutonomousODEHeunMethod
from StopConditions.StopAtPlateau import StopAtPlateau


def bungee_jumper_velocity(m, cd, t, g=9.8):
    return np.sqrt(g * m / cd) * np.tanh(np.sqrt(g*cd/m) * t)

def bungee_jumper_acceleration(m, cd, v, g=9.8):
    return g - (cd * v ** 2 /m)


def bungee_jump_case_study():
    m = 68.1  # kg
    cd = 0.25 # kg/m
    v0 = 0    # m/s
    dt = 2    # s

    bungee_acceleration_func = partial(bungee_jumper_acceleration, m, cd)

    # --------------- Analytical solution ---------------
    bungee_jumper_velocity_func = partial(bungee_jumper_velocity, m, cd)
    analytical_df = pd.DataFrame(
        data=dict(
            t=np.arange(0, 20, dt),
            x=bungee_jumper_velocity_func(np.arange(0, 20, dt))
        )
    )
    analytical_df.to_json(str(Path(__file__).parent / 'data' / 'bungeeJumper_analytical.json'))
    analytical_df.to_csv(str(Path(__file__).parent / 'data' / 'bungeeJumper_analytical.csv'))
    # --------------- Euler Method ---------------
    euler_solver = FirstOrderAutonomousODEEulerMethod(
        derivative_function=bungee_acceleration_func, t0=0, x0=v0, dt=dt,
        stop_conditions=[StopAtPlateau(tracking='x', patience=2)])
    euler_solver.run()
    euler_solver.export_history(str(Path(__file__).parent / 'data' / 'EulerMethod_bungeeJump.json'))
    euler_solver_df = euler_solver.error_analysis(analytic_solution_function=bungee_jumper_velocity_func)
    euler_solver_df.to_csv(str(Path(__file__).parent / 'data' / 'EulerMethod_bungeeJump_error.csv'))
    df_to_latex(
        euler_solver_df,
        str(Path(__file__).parent / 'data' / 'EulerMethod_bungeeJumper.tex'),
        formatting=dict(
            t=dict(header='$t$ (s)'),
            x=dict(header='$v$ (m/s)', format='0.6f'),
            dt=dict(header=r'$\Delta t$ (s)'),
            dx_dt=dict(header=r'$a (m/s^2)$', format='0.6f'),
            absolute_error=dict(header='$|e|$ (m/s)', format='0.6f'),
            relative_error=dict(header=r'$|\frac{e}{v_{\text{exact}}}|$ (m/s)\%',
                                format=lambda x: f'{x*100:0.6f}\%' if not pd.isna(x) else '-'),
        )
    )

    # --------------- Heuns Method --------------
    heun_solver = FirstOrderAutonomousODEHeunMethod(
        derivative_function=bungee_acceleration_func, t0=0, x0=v0, dt=dt,
        stop_conditions=[StopAtPlateau(tracking='x', patience=2)]
    )
    heun_solver.run()
    heun_solver.export_history(str(Path(__file__).parent / 'data' /  'HeunMethod_bungeeJump.json'))
    heun_solver.export_to_latex(str(Path(__file__).parent / 'data' / 'HeunMethod_bungeeJumper.tex'))
    heun_solver_df = heun_solver.error_analysis(analytic_solution_function=bungee_jumper_velocity_func)
    heun_solver_df.to_csv(str(Path(__file__).parent / 'data' / 'HeunMethod_bungeeJump_error.csv'))
    df_to_latex(
        heun_solver_df,
        str(Path(__file__).parent / 'data' / 'HeunMethod_bungeeJumper.tex'),
        formatting=dict(
            t=dict(header='$t$ (s)'),
            x=dict(header='$v$ (m/s)', format='0.6f'),
            dt=dict(header='$dt$ (s)'),
            dx_dt=dict(header=r'$a (m/s^2)$', format='0.6f'),
            predictor_x=dict(header='$v_{predictor}$ (m/s)', format='0.6f'),
            absolute_error=dict(header='$e$ (m/s)', format='0.6f'),
            relative_error=dict(header=r'$|\frac{e}{v_{\text{exact}}}|$ (m/s)\%',
                                format=lambda x: f'{x * 100:0.6f}\%' if not pd.isna(x) else '-'),
        )
    )

    # --------------- Plot ---------------
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.grid()
    ax.set_title('Bungee Jumper')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')

    ax.plot(analytical_df['t'], analytical_df['x'], color='red', label='Analytical solution')

    euler_solver.plot_history(x_var='t', y_var='x', ax=ax, color='blue',
                        label='Euler method', linestyle=':', marker='o')

    heun_solver.plot_history(x_var='t', y_var='x', ax=ax, color='green',
                         label='Heun method', linestyle='--', marker='o')

    ax.legend()
    ax.set_xlim(0, 20)

    plt.savefig(str(Path(__file__).parent / 'bungeeJump.png'))





if __name__ == '__main__':
    bungee_jump_case_study()