import os
import time
import numpy as np
from skopt import Optimizer
from skopt.space import Real, Integer
import sim_optimizer_couple as sim_optimizer
from sim_optimizer_couple import MAGNETIC_MOMENT, MAGNETIC_FIELD_MAGNITUDE
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
        import sim_optimizer_couple
        trajectory = sim_optimizer_couple.run_simulation(
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
SIM_DURATION = 5.0  # Simulation time per run (seconds). Shorter = faster iterations, noisier cost.
SIMULATION_TIMEOUT = 20  # Wall-clock s per worker. scene4 (4-legged) is heavier; bump to 30+ if it still times out.
ROLLOUTS_PER_SCENE = 1  # Sims per scene per iteration. Use 1 when sim is deterministic (same params → same cost); >1 only helps with noisy sims.
# 16 cores: use BATCH_SIZE=8 so 8×2=16 tasks = one full wave (~18s/batch). BATCH_SIZE=10 would give 20 tasks → 16+4 waves (~25s).
BATCH_SIZE = 8
# One scene per process; cap at core count so we don't oversubscribe.
NUM_SCENES = len(MJCF_PATHS)
POOL_SIZE = min(os.cpu_count() or 16, BATCH_SIZE * NUM_SCENES)  # cap at cores; 16 cores → 16 workers, 16 tasks/batch
VERBOSE_BATCH = True  # If True, print cost / speed residual (cm/s) / tumble for each of the N points per batch.
PROFILE_BATCH = True  # If True, print per-batch timing: ask, sim, aggregate, tell, csv (and verbose if VERBOSE_BATCH).
# Surrogate: "gp" = Gaussian process (ask/tell blow up with n, ~30s by batch 20). "rf" or "et" = random/extra trees (fast, ask/tell stay ~0.1s).
BASE_ESTIMATOR = "rf"

# Seed from older single-scene opt (optimized_params/optimization_results.csv row abe1b74c). Fewer params there; assume fudges=1, damping=7e-10.
SEED_FROM_OLD_CSV = False
# Order: sliding, torsional, rolling, solref_tc, solref_dr, solimp_dmin, solimp_dmax, solimp_width, moment_fudge, field_fudge, dof_damping
SEED_POINT = [
    0.00014225746640521907, 0.0021388784110800154, 5.292387847485097e-05,
    0.001, 0.7414912155887285, 0.9084351427617432, 0.9734506827063522, 0.0037927813470769885,
    1.0, 1.0, 7e-10,
]

# --- Cost function constants (tune these to change what the optimizer cares about) ---
COST_SETTLE_TIME = 0.1  # Time after which we measure velocity (s). Must match sim_optimizer_couple.SETTLE_TIME.
TUMBLE_THRESHOLD = 0.3  # cos(angle) below this = "tumbling". ~0.3 ≈ 72.5° from vertical.
TUMBLE_PENALTY_SCALE = 0.1  # Per-frame penalty when uprightness < threshold: (1 - uprightness) * this.
COST_FAILURE = 1e6  # Cost when simulation fails or trajectory is empty.
VELOCITY_COST_WEIGHT = 1.0  # Weight for (avg_velocity - target)².
TUMBLE_COST_WEIGHT = 1.0  # Weight for tumble penalty sum.

# This list will store detailed results from each trial
all_results = []
pool = None  # Global variable to hold the multiprocessing pool

CSV_PATH = "multi_optimization_results.csv"


def _csv_fieldnames():
    param_names = [dim.name for dim in space]
    scene_cost_names = [f"cost_{scene}" for scene in MJCF_PATHS.keys()]
    scene_vel_names = [f"velocity_{scene}" for scene in MJCF_PATHS.keys()]
    return ["id", "cost"] + scene_vel_names + scene_cost_names + param_names


def _append_result_to_csv(res):
    """Append one result row to the CSV. Call after each iteration."""
    row = {"id": res["id"], "cost": res["cost"]}
    for scene in MJCF_PATHS.keys():
        row[f"velocity_{scene}"] = res["scene_avg_velocities"].get(scene, 0)
        row[f"cost_{scene}"] = res["scene_costs"].get(scene, 0)
    row.update(res["params"])
    try:
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_csv_fieldnames())
            writer.writerow(row)
    except Exception as e:
        print(f"  [Warning] Could not append to CSV: {e}")


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
    Real(0.5, 1.5, "uniform", name='magnetic_moment_fudge'),
    Real(0.5, 1.5, "uniform", name='magnetic_field_fudge'),
    Real(7e-11, 7e-9, "log-uniform", name='dof_damping'),
]

def calculate_cost(trajectory, target_velocity, verbose=True):
    """
    Calculates a cost based on simulation trajectory, penalizing instability
    and rewarding consistent forward progress towards a target velocity.
    Returns a dictionary with detailed metrics.
    """
    if not trajectory:
        return {'total_cost': COST_FAILURE, 'avg_forward_velocity': 0, 'tumble_penalty': 0}

    # --- 1. Forward Velocity Cost (over active time) ---
    final_state = trajectory[-1]
    start_state = trajectory[0]
    for state in trajectory:
        if state['time'] >= COST_SETTLE_TIME:
            start_state = state
            break

    active_duration = final_state['time'] - start_state['time']
    avg_forward_velocity = 0
    if active_duration > 1e-6:
        forward_displacement = final_state['pos'][0] - start_state['pos'][0]
        avg_forward_velocity = forward_displacement / active_duration

    velocity_error = (avg_forward_velocity - target_velocity) ** 2

    # --- 2. Stability Cost (Tumbling Penalty) ---
    tumble_penalty = 0
    UP_VECTOR = np.array([0, 0, 1])
    for state in trajectory:
        quat = state['quat']
        body_z_axis = R.from_quat(quat).apply([0, 0, 1])
        uprightness = np.dot(body_z_axis, UP_VECTOR)
        if uprightness < TUMBLE_THRESHOLD:
            tumble_penalty += (1 - uprightness) * TUMBLE_PENALTY_SCALE

    total_cost = VELOCITY_COST_WEIGHT * velocity_error + TUMBLE_COST_WEIGHT * tumble_penalty

    if verbose:
        print(
            f"    Avg Vel: {avg_forward_velocity:.3f} m/s | "
            f"Vel Err: {velocity_error:.4f} | "
            f"Tumble Pen: {tumble_penalty:.4f} | "
            f"Total Cost: {total_cost:.4f}"
        )

    return {'total_cost': total_cost, 'avg_forward_velocity': avg_forward_velocity, 'tumble_penalty': tumble_penalty}


def _point_to_params(point):
    """Convert one point (list of values in space order) to params dict."""
    return {dim.name: point[i] for i, dim in enumerate(space)}


def _sim_params_from_point(point):
    """Build sim_params dict from one point (list). Same shape as _evaluate_point used."""
    params = _point_to_params(point)
    m_mag = MAGNETIC_MOMENT * params['magnetic_moment_fudge']
    kp_mag = m_mag * MAGNETIC_FIELD_MAGNITUDE * params['magnetic_field_fudge']
    return {
        'ground_friction': [params['sliding_friction'], params['torsional_friction'], params['rolling_friction']],
        'solref': [params['solref_timeconst'], params['solref_dampratio']],
        'solimp': [params['solimp_dmin'], params['solimp_dmax'], params['solimp_width'], 0.5, 1.0],
        'dof_damping': params['dof_damping'],
        'kp_mag': kp_mag,
        'mag_params': {'m_mag': m_mag},
    }


def _evaluate_one_scene(args):
    """
    Run one scene for one point. Used by 1-scene-per-process pool.
    args: (point_index, point, scene_name, mjcf_path)
    Returns: (point_index, scene_name, cost, velocity, tumble, wall_time)
    """
    point_index, point, scene_name, mjcf_path = args
    import sim_optimizer_couple as _sim
    sim_params = _sim_params_from_point(point)
    target_velocity = TARGET_VELOCITIES[scene_name]
    t0 = time.perf_counter()
    trajectory = _sim.run_simulation(sim_params, mjcf_path=mjcf_path, sim_duration=SIM_DURATION, visualize=False)
    wall_time = time.perf_counter() - t0
    if trajectory is None:
        cost_data = {'total_cost': COST_FAILURE, 'avg_forward_velocity': 0.0, 'tumble_penalty': 0.0}
    else:
        cost_data = calculate_cost(trajectory, target_velocity, verbose=False)
    return (point_index, scene_name, cost_data['total_cost'], cost_data['avg_forward_velocity'],
            cost_data.get('tumble_penalty', 0.0), wall_time)


def _aggregate_scene_results(points, scene_results):
    """Turn list of (point_index, scene_name, cost, velocity, tumble, wall_time) into list of full result dicts (one per point)."""
    from collections import defaultdict
    by_point = defaultdict(lambda: {"scene_costs": {}, "scene_avg_velocities": {}, "scene_tumble": {}, "scene_wall_times": []})
    for (point_index, scene_name, cost, velocity, tumble, wall_time) in scene_results:
        by_point[point_index]["scene_costs"][scene_name] = cost
        by_point[point_index]["scene_avg_velocities"][scene_name] = velocity
        by_point[point_index]["scene_tumble"][scene_name] = tumble
        by_point[point_index]["scene_wall_times"].append(wall_time)
    results = []
    for point_index in sorted(by_point.keys()):
        d = by_point[point_index]
        params = _point_to_params(points[point_index])
        total_cost = sum(d["scene_costs"].values())
        # Per-point "time" = max(scene times) since the two scenes run in parallel (different workers)
        point_wall = max(d["scene_wall_times"]) if d["scene_wall_times"] else 0.0
        results.append({
            'id': str(uuid.uuid4().hex)[:8],
            'cost': total_cost,
            'params': params,
            'scene_costs': d["scene_costs"],
            'scene_avg_velocities': d["scene_avg_velocities"],
            'scene_tumble': d["scene_tumble"],
            'wall_time': point_wall,
        })
    return results


# 2. Batch BO: optimizer proposes BATCH_SIZE points; we run 1 scene per process (BATCH_SIZE * num_scenes tasks in parallel).
def _run_batch_optimization():
    global pool
    optimizer = Optimizer(
        dimensions=space,
        base_estimator=BASE_ESTIMATOR,
        n_initial_points=20,
        random_state=42,
    )
    n_done = 0
    # Optional: seed with one point from older opt (evaluated in main process, no pool yet)
    if SEED_FROM_OLD_CSV:
        print("\n--- Seed from old CSV (optimization_results.csv abe1b74c, fudges=1, damping=7e-10) ---")
        scene_results = [_evaluate_one_scene((0, SEED_POINT, sn, path)) for sn, path in MJCF_PATHS.items()]
        seed_results = _aggregate_scene_results([SEED_POINT], scene_results)
        seed_result = seed_results[0]
        optimizer.tell([SEED_POINT], [seed_result["cost"]])
        all_results.append(seed_result)
        _append_result_to_csv(seed_result)
        n_done = 1
        sav = seed_result["scene_avg_velocities"]
        st = seed_result.get("scene_tumble", {})
        res_str = " | ".join(f"{s}: {(sav.get(s, 0) - TARGET_VELOCITIES[s]) * 100:+.1f} cm/s" for s in MJCF_PATHS.keys())
        tum_str = " | ".join(f"{s}: {st.get(s, 0.0):.4f}" for s in MJCF_PATHS.keys())
        print(f"  Seed cost={seed_result['cost']:.6f} | residual: {res_str} | tumble: {tum_str} | id={seed_result['id']}\n")
    pool = multiprocessing.Pool(processes=POOL_SIZE)
    batch_num = 0
    try:
        while n_done < N_CALLS:
            n_this = min(BATCH_SIZE, N_CALLS - n_done)
            batch_num += 1
            print(f"\n--- Batch {batch_num}: asking for {n_this} points ({n_done + 1}–{n_done + n_this} / {N_CALLS}), {n_this * NUM_SCENES} scene tasks ---")
            t_ask = time.perf_counter()
            points = optimizer.ask(n_points=n_this)
            if n_this == 1:
                points = [points]
            t_ask = time.perf_counter() - t_ask
            # One scene per process: (point_index, point, scene_name, mjcf_path)
            tasks = [
                (i, point, scene_name, mjcf_path)
                for i, point in enumerate(points)
                for scene_name, mjcf_path in MJCF_PATHS.items()
            ]
            t_sim = time.perf_counter()
            scene_results = pool.map(_evaluate_one_scene, tasks)
            t_sim = time.perf_counter() - t_sim
            t_agg = time.perf_counter()
            results = _aggregate_scene_results(points, scene_results)
            costs = [r["cost"] for r in results]
            t_agg = time.perf_counter() - t_agg
            t_tell = time.perf_counter()
            optimizer.tell(points, costs)
            t_tell = time.perf_counter() - t_tell
            t_csv = time.perf_counter()
            for r in results:
                all_results.append(r)
                _append_result_to_csv(r)
            t_csv = time.perf_counter() - t_csv
            n_done += n_this
            t_verbose = 0.0
            if VERBOSE_BATCH:
                t_verbose = time.perf_counter()
                for i, r in enumerate(results):
                    sav = r["scene_avg_velocities"]
                    st = r.get("scene_tumble", {})
                    residuals_cm_s = " | ".join(
                        f"{s}: {(sav.get(s, 0) - TARGET_VELOCITIES[s]) * 100:+.1f} cm/s"
                        for s in MJCF_PATHS.keys()
                    )
                    # tumble = per-scene tumble_penalty from calculate_cost; use .4f so small values visible
                    tumbles = " | ".join(f"{s}: {st.get(s, 0.0):.4f}" for s in MJCF_PATHS.keys())
                    wt = r.get("wall_time", 0)
                    print(f"    [{i+1}/{n_this}] cost={r['cost']:.4f} | residual: {residuals_cm_s} | tumble: {tumbles} | time: {wt:.1f}s")
                t_verbose = time.perf_counter() - t_verbose
            batch_wall = t_ask + t_sim + t_agg + t_tell + t_csv + t_verbose
            print(f"  Batch wall: {batch_wall:.1f}s | Costs: min={min(costs):.4f}, max={max(costs):.4f}")
            if PROFILE_BATCH:
                # tell = GP fit (grows with n_observations); use 3 decimals so early batches show e.g. 0.001s
                parts = f"ask={t_ask:.3f}s sim={t_sim:.2f}s agg={t_agg:.3f}s tell={t_tell:.3f}s csv={t_csv:.3f}s"
                if VERBOSE_BATCH:
                    parts += f" verbose={t_verbose:.3f}s"
                print(f"  Profile: {parts}")
            # Updated best trial so far (after this batch)
            best_so_far = min(all_results, key=lambda r: r["cost"])
            sav = best_so_far["scene_avg_velocities"]
            st = best_so_far.get("scene_tumble", {})
            res_str = " | ".join(f"{s}: {(sav.get(s, 0) - TARGET_VELOCITIES[s]) * 100:+.1f} cm/s" for s in MJCF_PATHS.keys())
            tum_str = " | ".join(f"{s}: {st.get(s, 0.0):.4f}" for s in MJCF_PATHS.keys())
            print(f"  Best so far (n={n_done}): cost={best_so_far['cost']:.6f} | residual: {res_str} | tumble: {tum_str} | id={best_so_far['id']}")
    finally:
        if pool:
            pool.terminate()
            pool.join()
            pool = None
    best = min(all_results, key=lambda r: r["cost"])
    return type("Result", (), {"fun": best["cost"], "x": [best["params"][dim.name] for dim in space]})()


if __name__ == "__main__":
    # 3. Run the optimization (batch BO: BATCH_SIZE points in parallel per batch).
    print(f"Running Bayesian optimization for {N_CALLS} evaluations in batches of {BATCH_SIZE}...")
    print(f"Target velocities: {TARGET_VELOCITIES}")

    # Write CSV header once (rows are appended after each batch)
    try:
        with open(CSV_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_csv_fieldnames()).writeheader()
    except Exception as e:
        print(f"  [Warning] Could not create CSV: {e}")

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    try:
        result = _run_batch_optimization()
    finally:
        if pool:
            print("\n--- Finalizing: Terminating worker pool. ---")
            pool.terminate()
            pool.join()
            pool = None

    # 4. Results were appended to CSV after each batch (in evaluation order)
    if all_results:
        print(f"\n--- Results appended to {CSV_PATH} after each batch (rows in evaluation order) ---")


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
        
        m_mag = MAGNETIC_MOMENT * result_data['params']['magnetic_moment_fudge']
        kp_mag = m_mag * MAGNETIC_FIELD_MAGNITUDE * result_data['params']['magnetic_field_fudge']
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
            'dof_damping': result_data['params']['dof_damping'],
            'kp_mag': kp_mag,
            'mag_params': {'m_mag': m_mag},
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
