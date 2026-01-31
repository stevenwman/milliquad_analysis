import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import sim_optimizer as sim_optimizer
from scipy.spatial.transform import Rotation as R
import csv
import uuid
import multiprocessing

# --- Worker Pool Function ---

# This function must be at the top level of the module to be pickleable by multiprocessing.
def run_simulation_worker(sim_params, mjcf_path, sim_duration):
    """
    A self-contained worker function that runs a simulation.
    This will be executed by a process in the multiprocessing pool.
    """
    try:
        # We re-import the module here because this runs in a separate process
        # that doesn't share memory with the main script.
        import sim_optimizer
        trajectory = sim_optimizer.run_simulation(
            sim_params,
            mjcf_path=mjcf_path,
            sim_duration=sim_duration,
            visualize=False
        )
        return trajectory
    except Exception as e:
        # Return the exception so the main process knows something went wrong.
        return e

# --- Optimization Configuration ---
TARGET_VELOCITIES = {
    "scene4": 0.21,  # 21 cm/s for 4-legged robot
    "scene2": 0.14   # 14 cm/s for 2-legged robot
}
MJCF_PATHS = {
    "scene4": "mulit_milli_quad/scene_4.xml",
    "scene2": "mulit_milli_quad/scene_2.xml"
}
N_CALLS = 200  # Number of optimization iterations
SIMULATION_TIMEOUT = 5  # seconds per simulation run

# This list will store detailed results from each trial
all_results = []
iteration_count = 0
pool = None # Global variable to hold the multiprocessing pool

# 1. Define the search space for the parameters.
# MuJoCo friction is [sliding, torsional, rolling].
# solref is [timeconst, dampratio].
# solimp is [dmin, dmax, width, midpoint, power] - we'll tune the first 3.
space = [
    # A friction value of 1.0 is very high and often causes the robot to
    # get stuck instantly. Reducing the upper bound makes the search more
    # efficient by focusing on more physically plausible values.
    Real(1e-5, 0.8, "log-uniform", name='sliding_friction'),
    Real(1e-5, 0.1, "log-uniform", name='torsional_friction'),
    Real(1e-5, 0.1, "log-uniform", name='rolling_friction'),
    Real(0.001, 0.1, "uniform", name='solref_timeconst'),
    Real(0.1, 2.0, "uniform", name='solref_dampratio'),
    Real(0.8, 0.99, "uniform", name='solimp_dmin'),
    Real(0.95, 0.999, "uniform", name='solimp_dmax'),
    Real(1e-4, 1e-2, "log-uniform", name='solimp_width'),
    Real(1e-7, 1e-5, "log-uniform", name='kp_mag')
]

def calculate_cost(trajectory, target_velocity):
    """
    Calculates a cost based on simulation trajectory, penalizing instability
    and rewarding consistent forward progress towards a target velocity.
    Returns a dictionary with detailed metrics.
    """
    if not trajectory:
        return {'total_cost': 1e6, 'avg_forward_velocity': 0}

    # --- 1. Forward Velocity Cost (over active time) ---
    final_state = trajectory[-1]
    settle_time = 0.1 # Must match the value in sim_optimizer.py

    # Find the starting state after the settling period
    start_state = trajectory[0]
    for state in trajectory:
        if state['time'] >= settle_time:
            start_state = state
            break

    active_duration = final_state['time'] - start_state['time']
    
    avg_forward_velocity = 0
    if active_duration > 1e-6: # Avoid division by zero
        # Forward displacement is distance traveled on x-axis during active time
        forward_displacement = final_state['pos'][0] - start_state['pos'][0]
        avg_forward_velocity = forward_displacement / active_duration
    
    velocity_error = (avg_forward_velocity - target_velocity)**2

    # --- 2. Stability Cost (Tumbling Penalty) ---
    tumble_penalty = 0
    UP_VECTOR = np.array([0, 0, 1])
    TUMBLE_THRESHOLD = 0.3 # Corresponds to a tilt of ~72.5 degrees

    for state in trajectory:
        quat = state['quat']
        body_z_axis = R.from_quat(quat).apply([0, 0, 1])
        uprightness = np.dot(body_z_axis, UP_VECTOR)
        
        if uprightness < TUMBLE_THRESHOLD:
            tumble_penalty += (1 - uprightness) * 0.1 

    total_cost = velocity_error + tumble_penalty
    
    print(
        f"    Avg Vel: {avg_forward_velocity:.3f} m/s | "
        f"Vel Err: {velocity_error:.4f} | "
        f"Tumble Pen: {tumble_penalty:.4f} | "
        f"Total Cost: {total_cost:.4f}"
    )

    return {'total_cost': total_cost, 'avg_forward_velocity': avg_forward_velocity}


# 2. Define the objective function to minimize.
# It takes the parameters, runs the simulation, and returns the error.
@use_named_args(space)
def objective(**params):
    """
    Objective function for the Bayesian optimizer.
    """
    global iteration_count, pool
    iteration_count += 1
    print(f"\n--- Iteration {iteration_count}/{N_CALLS} ---")

    # If the pool was terminated in a previous run, recreate it.
    if pool is None:
        print("  [Pool Info] Creating new worker pool.")
        pool = multiprocessing.Pool(processes=2)

    sim_params = {
        'ground_friction': [
            params['sliding_friction'],
            params['torsional_friction'],
            params['rolling_friction']
        ],
        'solref': [params['solref_timeconst'], params['solref_dampratio']],
        'solimp': [
            params['solimp_dmin'],
            params['solimp_dmax'],
            params['solimp_width'],
            0.5,  # default midpoint
            1.0   # default power
        ],
        # Keep other params constant for now
        'dof_damping': 7e-10,
        'kp_mag': params['kp_mag'],
        'mag_params': {
            # For now we keep these fixed. If you want to tune them, add
            # corresponding dimensions to `space` and wire them through here.
            'm_mag': 1.0,
            'k_int': 0.0,
            'mu0_over_4pi': 1e-7,
            'r_eps': 1e-4,
        },
    }

    total_cost = 0
    scene_costs = {}
    scene_avg_velocities = {}
    
    # Use the global worker pool to run simulations in parallel.
    async_results = {}
    for scene, mjcf_path in MJCF_PATHS.items():
        async_results[scene] = pool.apply_async(
            run_simulation_worker,
            (sim_params, mjcf_path, 5.0) # 5.0 is sim_duration
        )
        
    pool_needs_reset = False
    for scene, async_result in async_results.items():
        try:
            # Retrieve the result with a timeout.
            trajectory = async_result.get(timeout=SIMULATION_TIMEOUT)
            
            # Check if the worker returned an exception.
            if isinstance(trajectory, Exception):
                print(f"    [Error] Simulation for '{scene}' raised an exception: {trajectory}")
                trajectory = None

        except multiprocessing.TimeoutError:
            print(f"    [Timeout] Simulation for '{scene}' exceeded {SIMULATION_TIMEOUT}s.")
            trajectory = None
            pool_needs_reset = True
        
        target_velocity = TARGET_VELOCITIES[scene]
        
        if trajectory is None:
            print(f"    Simulation failed or timed out for {scene}. Assigning large penalty.")
            cost_data = {'total_cost': 1e6, 'avg_forward_velocity': 0}
        else:
            cost_data = calculate_cost(trajectory, target_velocity)

        total_cost += cost_data['total_cost']
        scene_costs[scene] = cost_data['total_cost']
        scene_avg_velocities[scene] = cost_data['avg_forward_velocity']

    # If a timeout occurred, the pool is now "dirty" with zombie workers.
    # Terminate it so a fresh one will be created on the next iteration.
    if pool_needs_reset:
        print("  [Pool Reset] Terminating dirty worker pool.")
        pool.terminate()
        pool.join()
        pool = None

    # Store detailed results for this run
    run_id = str(uuid.uuid4().hex)[:8]
    all_results.append({
        'id': run_id,
        'cost': total_cost,
        'params': params,
        'scene_costs': scene_costs,
        'scene_avg_velocities': scene_avg_velocities
    })

    print(f"  Finished run {run_id}. Total combined cost: {total_cost:.4f}")
    return total_cost

if __name__ == "__main__":
    # 3. Run the optimization.
    print(f"Running Bayesian optimization for {N_CALLS} iterations...")
    print(f"Target velocities: {TARGET_VELOCITIES}")
    
    # Use the 'spawn' start method for cleaner process creation, which is
    # important for libraries like MuJoCo that have complex internal state.
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # The start method can only be set once.
        pass

    try:
        result = gp_minimize(
            objective,
            space,
            n_calls=N_CALLS,
            random_state=42,
            # The optimizer builds a model of the cost function. When it encounters
            # bad parameters (high cost), it learns to avoid that region and samples
            # from more promising areas. Increasing n_initial_points gives it a
            # better starting model.
            n_initial_points=20
        )
    finally:
        # Ensure the final worker pool is always cleaned up.
        if pool:
            print("\n--- Finalizing: Terminating worker pool. ---")
            pool.terminate()
            pool.join()

    # 4. Save results to a CSV file
    if all_results:
        # Sort the results by cost before saving
        sorted_results = sorted(all_results, key=lambda r: r['cost'])

        param_names = [dim.name for dim in space]
        scene_cost_names = [f"cost_{scene}" for scene in MJCF_PATHS.keys()]
        scene_vel_names = [f"velocity_{scene}" for scene in MJCF_PATHS.keys()]
        
        fieldnames = ['id', 'cost'] + scene_vel_names + scene_cost_names + param_names
        try:
            with open('multi_optimization_results.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for res in sorted_results:
                    row = {
                        'id': res['id'],
                        'cost': res['cost'],
                    }
                    for scene in MJCF_PATHS.keys():
                        row[f"velocity_{scene}"] = res['scene_avg_velocities'].get(scene, 0)
                        row[f"cost_{scene}"] = res['scene_costs'].get(scene, 0)

                    row.update(res['params'])
                    writer.writerow(row)
            print("\n--- Optimization results saved to multi_optimization_results.csv (sorted by cost) ---")
        except Exception as e:
            print(f"\n--- Could not save optimization results to CSV: {e} ---")


    print("\n--- Optimization Finished ---")
    best_cost = result.fun
    print(f"Lowest Cost Found: {best_cost:.6f}")
    print("Best Parameters:")
    for dim, value in zip(space, result.x):
        print(f"  {dim.name}: {value:.6f}")

    # --- Top 5 Rollouts ---
    print("\n--- Recording Top 5 Best Rollouts ---")

    sorted_results = sorted(all_results, key=lambda r: r['cost'])

    for i in range(min(5, len(sorted_results))):
        result_data = sorted_results[i]
        rank = i + 1

        print(f"\n#{rank}: Cost={result_data['cost']:.6f}")
        for scene, velocity in result_data['scene_avg_velocities'].items():
            print(f"  - {scene}: Avg Velocity={velocity:.4f} m/s (Cost: {result_data['scene_costs'].get(scene, 'N/A'):.4f})")
        
        sim_params = {
            'ground_friction': [
                result_data['params']['sliding_friction'],
                result_data['params']['torsional_friction'],
                result_data['params']['rolling_friction']
            ],
            'solref': [
                result_data['params']['solref_timeconst'],
                result_data['params']['solref_dampratio']
            ],
            'solimp': [
                result_data['params']['solimp_dmin'],
                result_data['params']['solimp_dmax'],
                result_data['params']['solimp_width'], 
                0.5, 
                1.0
            ],
            'dof_damping': 7e-10,
            'kp_mag': result_data['params']['kp_mag']
        }

        video_dir = "top_5_rollouts_multi"
        
        for scene, mjcf_path in MJCF_PATHS.items():
            # For the final recording, we can run it in the main process
            # as hangs should be very rare with the best parameters.
            # This simplifies cleanup and avoids potential issues with OpenGL contexts
            # in subprocesses during rendering.
            video_path = f"{video_dir}/rank_{rank}_{scene}_id_{result_data['id']}_cost_{result_data['cost']:.4f}.mp4"
            print(f"  Recording video for {scene} to {video_path}...")
            sim_optimizer.run_simulation(
                sim_params,
                mjcf_path=mjcf_path,
                sim_duration=10.0,
                record_path=video_path
            )
