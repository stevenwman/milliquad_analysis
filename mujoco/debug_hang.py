import multiprocessing
import time
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from sim_optimizer import run_simulation

# --- Worker and Timeout Configuration ---

SIMULATION_TIMEOUT = 5.0  # seconds

def simulation_worker(params, mjcf_path, sim_duration, result_queue):
    """
    Worker function to run a single simulation.
    Designed to be executed in a separate process to isolate hangs.
    """
    try:
        trajectory = run_simulation(
            params,
            mjcf_path=mjcf_path,
            sim_duration=sim_duration,
            visualize=False
        )
        result_queue.put(trajectory)
    except Exception as e:
        result_queue.put(e)

def run_simulation_with_timeout(params, mjcf_path, sim_duration):
    """
    Runs the simulation in a subprocess with a timeout.
    Returns the trajectory or None if it times out or fails.
    """
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=simulation_worker,
        args=(params, mjcf_path, sim_duration, result_queue)
    )
    process.start()
    process.join(timeout=SIMULATION_TIMEOUT)

    if process.is_alive():
        print("  [Timeout] Simulation exceeded time limit. Terminating.")
        process.terminate()
        process.join()
        return None
    
    result = result_queue.get()
    if isinstance(result, Exception):
        print(f"  [Error] Simulation process raised an exception: {result}")
        return None
    
    return result

# --- Bayesian Optimization Test Setup ---

# Define a search space that is likely to contain unstable values
space = [
    Real(0.5, 3.0, "uniform", name='sliding_friction'),
    Real(0.5, 3.0, "uniform", name='torsional_friction'),
    Real(0.5, 3.0, "uniform", name='rolling_friction'),
]

@use_named_args(space)
def objective(**params):
    """
    The objective function for gp_minimize.
    It calls the timeout-wrapped simulation runner.
    """
    print(f"\nTesting parameters: {params}")
    
    # Use a fixed set of other parameters for this test
    full_params = {
        'ground_friction': [
            params['sliding_friction'],
            params['torsional_friction'],
            params['rolling_friction']
        ],
        'solref': [0.02, 1],
        'solimp': [0.9, 0.95, 0.001, 0.5, 2],
        'kp_mag': 5e-6,
        'dof_damping': 1e-9,
    }

    trajectory = run_simulation_with_timeout(
        full_params,
        mjcf_path="mulit_milli_quad/scene_4.xml",
        sim_duration=5.0
    )

    if trajectory is None:
        print("  -> Simulation failed or timed out. Assigning high cost.")
        return 1e6  # Return a large penalty

    # A simple cost function for successful runs
    final_pos = trajectory[-1]['pos'][0]
    cost = -final_pos  # We want to maximize forward distance
    print(f"  -> Simulation succeeded. Final position: {final_pos:.4f}, Cost: {cost:.4f}")
    return cost

if __name__ == "__main__":
    print("--- Testing gp_minimize with Process-Based Timeout ---")
    print(f"Each simulation call is limited to {SIMULATION_TIMEOUT} seconds.")
    print("The optimizer will now run for a few iterations. Expect some to time out.")

    result = gp_minimize(
        objective,
        space,
        n_calls=5, # Run a few test iterations
        random_state=42,
        n_initial_points=2
    )

    print("\n--- Test Complete ---")
    print(f"gp_minimize finished running. Best cost found: {result.fun:.4f}")
    print("This confirms that the optimizer can gracefully handle simulation timeouts.")
