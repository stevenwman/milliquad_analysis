# solver.py
import numpy as np
from scipy.integrate import solve_ivp
import pickle

def wrap_to_pi(x):
    """Wraps angles to the interval [-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi

def model(t, y, params):
    """
    Defines the differential equations for the coupled pendulum system.
    
    y[0] = x1 (theta1)
    y[1] = x2 (theta1_dot)
    y[2] = x3 (theta2)
    y[3] = x4 (theta2_dot)
    """
    # Unpack parameters
    B = params['B']
    wb = params['wb']
    c1 = params['c1']
    c2 = params['c2']
    g = params['g']
    l = params['l']
    J = params['J']
    m_pen = params['m_pen']
    B_inf = params['B_inf']
    Ff = params['Ff']
    x_range = params['x_range']
    x_offset = params['x_offset']
    cf = params['cf']

    # Unpack states
    x1, x2, x3, x4 = y
    
    # Calculate resistive forces
    # Note: The MATLAB code had -Ff * ... * x2.
    # F_res1 = Ff * np.exp(-((x1 - x_offset) * x_range)**2)

    theta = x1
    omega = x2

    f_hat = (-1.180609e-07 \
    - 2.246739e-06*theta \
    + 1.775195e-09*omega \
    + 2.778458e-06*theta**2 \
    - 1.155633e-08*theta*omega \
    - 7.178837e-07*theta**3 \
    + 3.862178e-09*theta**2*omega \
    - 1.286576e-07*theta**4 \
    + 8.003569e-09*theta**3*omega \
    + 6.522879e-08*theta**5 \
    - 3.200381e-09*theta**4*omega \
    - 5.847875e-09*theta**6)

    F_res1 = f_hat / 100

    F_res2 = 0.0
    
    # Calculate derivatives
    x1d = x2
    x3d = x4
    
    # Note: np.cos(np.pi - x) is equivalent to -np.cos(x)
    # The gravity term is -m_pen * g * l * np.cos(np.pi - x1)
    # This simplifies to +m_pen * g * l * np.cos(x1)
    # We translate the original MATLAB logic directly.
    
    # x2d = (B * np.sin(wrap_to_pi(wb * t - x1)) +
    #        B_inf * np.sin(wrap_to_pi(x3 - x1)) -
    #        c1 * x2 -
    #        m_pen * g * l * np.cos(np.pi - x1) +
    #        - F_res1 - c_res1 * x2) / J

    x2d = (B * np.sin((wb * t - x1)) +
           F_res1) / J
           
    x4d = (B * np.sin((wb * t - x3)) +
           B_inf * np.sin((x1 - x3)) -
           c2 * x4 -
           m_pen * g * l * np.cos(np.pi - x3) +
           F_res2) / J
           
    return [x1d, x2d, x3d, x4d]

def run_simulation():
    """
    Sets up and runs the ODE simulation, saving the results to a file.
    """
    # --- Parameters ---
    c1 = 3.22e-9
    c2 = 1 * c1
    m = 1.4e-5
    m_pen = m / 4
    d = 2e-3
    J = 1/6 * m * d**2
    B = 1e-6
    wb = 10 * 2 * np.pi
    g = 9.81
    l = 1e-3  # Pendulum length
    B_inf = 1e-7 * 0
    Ff = 1e-6
    cf = 1e-7
    x_range = 4 * np.pi
    x_offset = np.pi / 4

    # Pack parameters into a dictionary
    params = {
        'Ff': Ff, 'x_range': x_range, 'x_offset': x_offset, 'B': B, 'wb': wb,
        'c1': c1, 'c2': c2, 'g': g, 'l': l, 'J': J, 'm_pen': m_pen, 'cf': cf,
        'B_inf': B_inf
    }

    # --- Simulation Setup ---
    t_span = [0, 1]
    y0 = [0, 0, 0, 0]

    # --- Run Simulation ---
    print('Running simulation...')
    # We use 'BDF' as it's a stiff solver, analogous to MATLAB's ode15s.
    # dense_output=True is crucial as it creates an interpolator
    # which is used for smooth animation later.
    sol = solve_ivp(
        model, 
        t_span, 
        y0, 
        args=(params,), 
        method='BDF', 
        dense_output=True
    )
    print('Simulation finished.')

    # --- Save Results ---
    # We save the entire solution object 'sol' (which includes t, y, and
    # the interpolator) and the 'params' dictionary.
    results_data = {
        'sol': sol,
        'params': params
    }
    
    output_filename = 'simulation_data.pkl'
    with open(output_filename, 'wb') as f:
        pickle.dump(results_data, f)
        
    print(f'Simulation results saved to {output_filename}')

if __name__ == "__main__":
    run_simulation()