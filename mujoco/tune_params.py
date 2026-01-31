import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import sim_optimizer as sim_optimizer
from scipy.spatial.transform import Rotation as R
import csv
import uuid

# --- Optimization Configuration ---
TARGET_VELOCITY = 0.21  # 21 cm/s
N_CALLS = 20  # Number of optimization iterations

# This list will store detailed results from each trial
all_results = []


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
    # Real(0.0, 1e-5, "log-uniform", name='k_int'),  # coupling strength
    # Real(0.1, 10.0, "log-uniform", name='m_mag'),  # dipole moment
]

# --- Old Cost Function (for reference) ---
# def calculate_cost(trajectory, target_velocity):
#     """
#     Calculates a cost based on simulation trajectory, penalizing instability
#     and rewarding consistent forward progress towards a target velocity.
#     Returns a dictionary with detailed metrics.
#     """
#     if not trajectory:
#         return {'total_cost': 1e6, 'avg_forward_velocity': 0}
#
#     # --- 1. Forward Velocity Cost (over total time) ---
#     final_state = trajectory[-1]
#     duration = final_state['time']
#
#     avg_forward_velocity = 0
#     if duration > 0:
#         avg_forward_velocity = final_state['pos'][0] / duration
#
#     velocity_error = (avg_forward_velocity - target_velocity)**2
#
#     # --- 2. Stability Cost (Tumbling Penalty) ---
#     tumble_penalty = 0
#     UP_VECTOR = np.array([0, 0, 1])
#     TUMBLE_THRESHOLD = 0.3
#
#     for state in trajectory:
#         quat = state['quat']
#         body_z_axis = R.from_quat(quat).apply([0, 0, 1])
#         uprightness = np.dot(body_z_axis, UP_VECTOR)
#
#         if uprightness < TUMBLE_THRESHOLD:
#             tumble_penalty += (1 - uprightness) * 0.1
#
#     total_cost = velocity_error + tumble_penalty
#
#     print(
#         f"  Avg Vel: {avg_forward_velocity:.3f} m/s | "
#         f"Vel Err: {velocity_error:.4f} | "
#         f"Tumble Pen: {tumble_penalty:.4f} | "
#         f"Total Cost: {total_cost:.4f}"
#     )
#
#     return {'total_cost': total_cost, 'avg_forward_velocity': avg_forward_velocity}


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
        f"  Avg Vel: {avg_forward_velocity:.3f} m/s | "
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
        'kp_mag': 2.5e-6,
        'mag_params': {
            # For now we keep these fixed. If you want to tune them, add
            # corresponding dimensions to `space` and wire them through here.
            'm_mag': 1.0,
            'k_int': 0.0,
            'mu0_over_4pi': 1e-7,
            'r_eps': 1e-4,
        },
    }

    trajectory = sim_optimizer.run_simulation(
        sim_params, sim_duration=5.0, debug=False
    )
    
    if trajectory is None:
        print("  Simulation unstable. Assigning large penalty.")
        cost_data = {'total_cost': 1e6, 'avg_forward_velocity': 0}
    else:
        cost_data = calculate_cost(trajectory, TARGET_VELOCITY)

    # Store detailed results for this run
    run_id = str(uuid.uuid4().hex)[:8]
    all_results.append({
        'id': run_id,
        'cost': cost_data['total_cost'],
        'avg_velocity': cost_data['avg_forward_velocity'],
        'params': params
    })

    return cost_data['total_cost']

if __name__ == "__main__":
    # 3. Run the optimization.
    print(f"Running Bayesian optimization for {N_CALLS} iterations...")
    print(f"Target velocity is {TARGET_VELOCITY} m/s.")
    
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

    # 4. Save results to a CSV file
    if all_results:
        # Sort the results by cost before saving
        sorted_results = sorted(all_results, key=lambda r: r['cost'])

        param_names = [dim.name for dim in space]
        fieldnames = ['id', 'cost', 'avg_velocity'] + param_names
        try:
            with open('optimization_results.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for res in sorted_results:
                    row = {
                        'id': res['id'],
                        'cost': res['cost'],
                        'avg_velocity': res['avg_velocity']
                    }
                    row.update(res['params'])
                    writer.writerow(row)
            print("\n--- Optimization results saved to optimization_results.csv (sorted by cost) ---")
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

        print(f"\n#{rank}: Cost={result_data['cost']:.6f}, Avg Velocity={result_data['avg_velocity']:.4f} m/s")
        
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
            'kp_mag': 2.5e-6
        }

        video_dir = "top_5_rollouts"
        video_path = f"{video_dir}/rank_{rank}_id_{result_data['id']}_cost_{result_data['cost']:.4f}.mp4"

        print(f"  Recording video to {video_path}...")

        sim_optimizer.run_simulation(
            sim_params,
            sim_duration=10.0,
            record_path=video_path
        )
