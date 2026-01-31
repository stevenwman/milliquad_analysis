"""
Parameter Sweep for Coupled Pendulum System
Performs 10x10 parameter sweep over Ff and cf (c1) parameters
Plots polar plots of the first pendulum for each parameter combination
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pickle
import os
from tqdm import tqdm
import time

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
    cf = params['cf']
    x_range = params['x_range']
    x_offset = params['x_offset']
    
    # Unpack states
    x1, x2, x3, x4 = y
    
    # Calculate resistive forces
    F_res1 = Ff * np.exp(-((x1 - x_offset) * x_range)**2)
    c_res1 = cf * np.exp(-((x1 - x_offset) * x_range)**2)
    F_res2 = 0.0
    
    # Calculate derivatives
    x1d = x2
    x3d = x4
    
    x2d = (B * np.sin(wrap_to_pi(wb * t - x1)) +
           B_inf * np.sin(wrap_to_pi(x3 - x1)) -
           c1 * x2 -
           m_pen * g * l * np.cos(np.pi - x1) +
           - F_res1 - c_res1 * x2) / J
           
    x4d = (B * np.sin(wrap_to_pi(wb * t - x3)) +
           B_inf * np.sin(wrap_to_pi(x1 - x3)) -
           c2 * x4 -
           m_pen * g * l * np.cos(np.pi - x3) +
           F_res2) / J
           
    return [x1d, x2d, x3d, x4d]

def run_single_simulation(Ff, cf, save_results=False):
    """
    Run a single simulation with given Ff and cf parameters.
    
    Parameters:
    -----------
    Ff : float
        Friction force parameter
    cf : float
        Friction coefficient parameter
    save_results : bool
        Whether to save individual simulation results
        
    Returns:
    --------
    t : array
        Time points
    theta1 : array
        First pendulum angle
    theta1_dot : array
        First pendulum angular velocity
    """
    # Base parameters (same as original solver.py)
    c1 = 3.22e-9  # Fixed damping coefficient
    c2 = 1 * c1  # c2 = c1
    m = 1.4e-5
    m_pen = m / 4
    d = 2e-3
    J = 1/6 * m * d**2
    B = 1e-6
    wb = 40 * 2 * np.pi
    g = 9.81
    l = 1e-3
    B_inf = 1e-7 * 0
    x_range = 4 * np.pi
    x_offset = np.pi / 4

    # Pack parameters into a dictionary
    params = {
        'Ff': Ff, 'cf': cf, 'x_range': x_range, 'x_offset': x_offset, 'B': B, 'wb': wb,
        'c1': c1, 'c2': c2, 'g': g, 'l': l, 'J': J, 'm_pen': m_pen,
        'B_inf': B_inf
    }

    # Simulation setup
    t_span = [0, 0.4]
    y0 = [0, 0, 0, 0]

    # Run simulation
    sol = solve_ivp(
        model, 
        t_span, 
        y0, 
        args=(params,), 
        method='BDF', 
        dense_output=True,
        rtol=1e-6,
        atol=1e-9
    )
    
    # Extract results
    t = sol.t
    theta1 = sol.y[0]
    theta1_dot = sol.y[1]
    
    if save_results:
        # Save individual simulation if requested
        results_data = {
            'sol': sol,
            'params': params,
            'Ff': Ff,
            'cf': cf
        }
        
        filename = f'simulation_Ff_{Ff:.2e}_cf_{cf:.2e}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(results_data, f)
    
    return t, theta1, theta1_dot

def create_polar_plot(t, theta1, theta1_dot, Ff, cf, save_path=None):
    """
    Create polar plot of the first pendulum trajectory with time-based color coding.
    
    Parameters:
    -----------
    t : array
        Time points
    theta1 : array
        First pendulum angle
    theta1_dot : array
        First pendulum angular velocity
    Ff : float
        Friction force parameter
    cf : float
        Friction coefficient parameter
    save_path : str, optional
        Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(projection='polar'))
    
    # Normalize time for color mapping (0 to 1)
    t_norm = (t - t[0]) / (t[-1] - t[0])
    
    # Polar plot with time-based color coding
    for i in range(len(theta1)-1):
        ax1.plot([theta1[i], theta1[i+1]], [theta1_dot[i], theta1_dot[i+1]], 
                color=plt.cm.viridis(t_norm[i]), linewidth=1)
    ax1.set_title(f'Polar Phase Space (Ff={Ff:.2e}, cf={cf:.2e})')
    ax1.grid(True)
    
    # Polar trajectory with time-based color coding
    r = np.abs(theta1_dot)
    theta = theta1
    
    for i in range(len(theta)-1):
        ax2.plot([theta[i], theta[i+1]], [r[i], r[i+1]], 
                color=plt.cm.viridis(t_norm[i]), linewidth=1)
    ax2.set_title(f'Polar Trajectory (Ff={Ff:.2e}, cf={cf:.2e})')
    ax2.set_ylim(0, np.max(r) * 1.1)
    ax2.grid(True)
    
    # Add colorbar to show time progression
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=t[0], vmax=t[-1]))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, shrink=0.8, pad=0.1)
    cbar.set_label('Time (s)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def parameter_sweep():
    """
    Perform 10x10 parameter sweep over Ff and c1 parameters.
    """
    # Create parameter ranges
    Ff_values = np.logspace(-6, -9, 6)  # 1e-6 to 1e-10
    Ff_values = np.append(Ff_values, 0)  # Add 0 as the lowest value
    cf_values = np.logspace(-7, -9, 6)  # 1e-8 to 1e-10
    cf_values = np.append(cf_values, 0)  # Add 0 as the lowest value
    
    print(f"Parameter sweep: {len(Ff_values)} x {len(cf_values)} = {len(Ff_values) * len(cf_values)} simulations")
    print(f"Ff range: {Ff_values[0]:.2e} to {Ff_values[-1]:.2e}")
    print(f"cf range: {cf_values[0]:.2e} to {cf_values[-1]:.2e}")
    
    # Create output directory
    output_dir = "parameter_sweep_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Store results for analysis
    results = []
    
    # Progress bar
    total_sims = len(Ff_values) * len(cf_values)
    pbar = tqdm(total=total_sims, desc="Running parameter sweep")
    
    start_time = time.time()
    
    for i, Ff in enumerate(Ff_values):
        for j, cf in enumerate(cf_values):
            try:
                # Run simulation
                t, theta1, theta1_dot = run_single_simulation(Ff, cf, save_results=False)
                
                # Create and save polar plot
                plot_filename = f"{output_dir}/polar_Ff_{Ff:.2e}_cf_{cf:.2e}.png"
                create_polar_plot(t, theta1, theta1_dot, Ff, cf, save_path=plot_filename)
                
                # Store results
                results.append({
                    'Ff': Ff,
                    'cf': cf,
                    't': t,
                    'theta1': theta1,
                    'theta1_dot': theta1_dot,
                    'plot_file': plot_filename
                })
                
            except Exception as e:
                print(f"Error with Ff={Ff:.2e}, cf={cf:.2e}: {e}")
                results.append({
                    'Ff': Ff,
                    'cf': cf,
                    'error': str(e)
                })
            
            pbar.update(1)
    
    pbar.close()
    
    # Save all results
    results_filename = f"{output_dir}/parameter_sweep_results.pkl"
    with open(results_filename, 'wb') as f:
        pickle.dump(results, f)
    
    elapsed_time = time.time() - start_time
    print(f"\nParameter sweep completed in {elapsed_time:.2f} seconds")
    print(f"Results saved to {results_filename}")
    print(f"Individual plots saved to {output_dir}/")
    
    return results

def create_panel_plot(results):
    """
    Create a 10x10 panel plot showing all polar plots in a grid.
    """
    # Extract successful results
    successful_results = [r for r in results if 'error' not in r]
    
    if not successful_results:
        print("No successful simulations to plot")
        return
    
    # Create parameter grid
    Ff_values = sorted(list(set([r['Ff'] for r in successful_results])))
    cf_values = sorted(list(set([r['cf'] for r in successful_results])))
    
    # Create subplot grid
    fig, axes = plt.subplots(len(cf_values), len(Ff_values), figsize=(20, 20), subplot_kw=dict(projection='polar'))
    
    for i, Ff in enumerate(Ff_values):
        for j, cf in enumerate(cf_values):
            # Find corresponding result
            result = next((r for r in successful_results if r['Ff'] == Ff and r['cf'] == cf), None)
            if result:
                ax = axes[j, i]  # Note: j,i for correct orientation
                
                # Create polar plot with time-based color coding
                r = np.abs(result['theta1_dot'])
                theta = result['theta1']
                t = result['t']
                
                # Normalize time for color mapping
                t_norm = (t - t[0]) / (t[-1] - t[0])
                
                # Plot with time-based colors
                for k in range(len(theta)-1):
                    ax.plot([theta[k], theta[k+1]], [r[k], r[k+1]], 
                           color=plt.cm.viridis(t_norm[k]), linewidth=0.5)
                
                ax.set_title(f'Ff={Ff:.1e}\ncf={cf:.1e}', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                # Set consistent limits
                if len(r) > 0:
                    ax.set_ylim(0, np.max(r) * 1.1)
            else:
                # Empty subplot for failed simulations
                axes[j, i].set_title(f'Ff={Ff:.1e}\ncf={cf:.1e}\n(Error)', fontsize=8, color='red')
    
    plt.tight_layout()
    plt.savefig('parameter_sweep_results/parameter_sweep_panel.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Starting parameter sweep...")
    results = parameter_sweep()
    create_panel_plot(results)
    print("Parameter sweep analysis complete!")
