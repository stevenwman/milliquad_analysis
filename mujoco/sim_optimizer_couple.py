import mujoco
import mujoco.viewer
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import imageio
import imageio.plugins.ffmpeg
import pathlib

# --- Simulation Constants ---
# These constants are used both here and in tuning scripts (tune_params.py, tune_multi_params.py)
# If you change them, update the tuning scripts accordingly.
SETTLE_TIME = 0.1  # Time before driving starts (seconds)
STUCK_CHECK_INTERVAL = 5.0  # How often to check if robot is stuck (seconds)
STUCK_THRESHOLD = 0.005  # Minimum movement distance to avoid "stuck" detection (meters)
SIM_TIMESTEP = 1.0 / 2000.0  # MuJoCo timestep (seconds)
VIDEO_FRAMERATE = 60.0  # Frames per second for video recording

MU0_OVER_4PI = 1e-7 # μ₀/(4π) in SI (N/A²). Dipole field B = (μ₀/4π) r⁻³ [3(m·r̂)r̂ − m]. No need to tune.
R_EPS = 1e-6 # Minimum r in dipole field to avoid 1/r³ blow-up when bodies are very close (meters).
MAGNETIC_MOMENT = 1.13e-3
MAGNETIC_FIELD_MAGNITUDE = 2e-3


def add_visual_arrow(scene, from_point, to_point, radius=0.001, rgba=(0, 0, 1, 1)):
    """
    Adds a single visual arrow to the mjvScene.
    This is a visual-only object and does not affect the physics.
    """
    if scene.ngeom >= scene.maxgeom:
        print("Warning: Maximum number of geoms reached. Cannot add arrow.")
        return
    
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, type=mujoco.mjtGeom.mjGEOM_ARROW,
                        size=np.array([radius, radius, np.linalg.norm(to_point - from_point)]),
                        pos=np.zeros(3), mat=np.eye(3).flatten(), # Will be updated by mjv_connector
                        rgba=np.array(rgba, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 
                         radius, from_point, to_point)
    scene.ngeom += 1

def add_text(data, viewer, text_input):
    """Adds text to the scene."""
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_LABEL,
        size=np.array([0.2, 0.2, 0.2]),
        pos=data.qpos[:3] + np.array([0.0, 0.0, 0.01]),
        mat=np.eye(3).flatten(),
        rgba=np.array([0, 0, 0, 0])  # invisible
    )
    geom.label = text_input
    viewer.user_scn.ngeom += 1


# --- Helper Functions for Simulation Logic ---

def _initialize_pose(data):
    """Initialize the robot pose. Main body at z=0.002, quaternion identity, legs at pi."""
    data.qpos[2] = 0.002
    data.qpos[3:7] = [0, 0, 1, 0]  # Main body quaternion (w, x, y, z)
    data.qpos[7:11] = np.pi * np.ones(4)  # Leg angles


def _compute_drive_angle(time, drive_freq, settle_time):
    """Compute the drive angle based on time. Returns 0 during settle period."""
    if time < settle_time:
        return 0.0
    return ((time - settle_time) * drive_freq * 2 * np.pi) % (2 * np.pi)


def _get_magnet_state(data, i):
    """Return (body_idx, pos_world, north_world_unit) for leg i."""
    body_idx = i + 2
    pos = data.xpos[body_idx].copy()
    quat = data.xquat[body_idx]
    Rwb = R.from_quat(quat, scalar_first=True).as_matrix()
    # joints 0 and 2 point in opposite direction than 1 and 3
    body_dir = np.array([1, 0, 0]) if i in [0, 2] else np.array([-1, 0, 0])
    north = Rwb @ body_dir
    norm = np.linalg.norm(north)
    if norm > 0:
        north /= norm
    return body_idx, pos, north


def _dipole_field(mj, r_vec):
    """Magnetic field B at offset r_vec from dipole moment mj. Uses MU0_OVER_4PI, R_EPS."""
    r = np.linalg.norm(r_vec)
    r = max(r, R_EPS)
    rhat = r_vec / r
    return MU0_OVER_4PI * (1.0 / r**3) * (3.0 * np.dot(mj, rhat) * rhat - mj)


def _compute_external_torques(data, angle, kp_mag, settle_time):
    """Return tau_ext[4,3] world torques from external drive (no side effects)."""
    tau_ext = np.zeros((4, 3))
    if data.time <= settle_time:
        return tau_ext

    world_frame_dir = np.array([0, 0, 1])
    goal = R.from_euler('y', angle, degrees=False).as_matrix() @ world_frame_dir
    norm = np.linalg.norm(goal)
    if norm > 0:
        goal /= norm

    for i in range(4):
        _, _, north = _get_magnet_state(data, i)
        tau_ext[i] = kp_mag * np.cross(north, goal)
    return tau_ext


def _compute_interjoint_torques(data, m_mag):
    """Return tau_int[4,3] world torques from dipole-dipole coupling (no side effects). τ = m × B."""
    tau_int = np.zeros((4, 3))
    if m_mag == 0.0:
        return tau_int

    # Gather states once
    pos = np.zeros((4, 3))
    north = np.zeros((4, 3))
    for i in range(4):
        _, pi, ni = _get_magnet_state(data, i)
        pos[i] = pi
        north[i] = ni

    m = m_mag * north  # dipole moments

    for i in range(4):
        Bi = np.zeros(3)
        for j in range(4):
            if j == i:
                continue
            Bi += _dipole_field(m[j], pos[i] - pos[j])
        tau_int[i] = np.cross(m[i], Bi)  # τ = m × B

    return tau_int


def _apply_magnetic_forces(model, data, kp_mag, drive_freq, settle_time, mag_params, step_cache):
    """
    Apply magnetic torques to 4 leg bodies based on external drive and inter-joint coupling.

    Side effects:
        - Writes data.xfrc_applied[:, 3:6] with total world torques.
        - Stores tau_ext, tau_int, angle into step_cache for later use (e.g. power).
    """
    # Compute drive angle
    angle = _compute_drive_angle(data.time, drive_freq, settle_time)

    # Always clear applied wrenches each step
    data.xfrc_applied[:, :] = 0.0

    # External torques from drive field
    tau_ext = _compute_external_torques(data, angle, kp_mag, settle_time)

    # Inter-joint torques from dipole-dipole coupling.
    # Be strict here: all required parameters must be provided explicitly so we
    # don't accidentally run with silent defaults.
    required_keys = ("m_mag",)
    for key in required_keys:
        if key not in mag_params:
            raise ValueError(
                f"mag_params missing required key '{key}'. "
                "Define it in the optimization loop / caller."
            )

    tau_int = _compute_interjoint_torques(data, m_mag=mag_params["m_mag"])

    # Apply total torque to bodies
    for i in range(4):
        body_idx, _, _ = _get_magnet_state(data, i)
        data.xfrc_applied[body_idx, 3:6] += (tau_ext[i] + tau_int[i])

    # Stash for trajectory / power accounting
    step_cache["tau_ext"] = tau_ext
    step_cache["tau_int"] = tau_int
    step_cache["angle"] = angle

    return angle  # For visualization


def _update_viewer_overlays(viewer, data, drive_freq, kp_mag, initial_pos, angle):
    """Update visual overlays (arrows and text) in the viewer."""
    viewer.user_scn.ngeom = 0
    
    # Draw arrows for each leg showing magnet direction and goal direction
    for i in range(4):
        body_idx = i + 2
        body_quat = data.xquat[body_idx]
        body_pos = data.xpos[body_idx]
        body_frame = R.from_quat(body_quat, scalar_first=True)
        
        body_frame_dir = np.array([1, 0, 0]) if i in [0, 2] else np.array([-1, 0, 0])
        world_frame_dir = np.array([0, 0, 1])
        
        magnet_north = body_frame.as_matrix() @ body_frame_dir
        rpy_rot = R.from_euler('y', angle, degrees=False)
        goal_north = rpy_rot.as_matrix() @ world_frame_dir
        
        arr_len = 0.01
        to = body_pos + arr_len * magnet_north
        to_goal = body_pos + arr_len * goal_north
        add_visual_arrow(viewer.user_scn, body_pos[:3], to, rgba=(0, 1, 0, 1))
        add_visual_arrow(viewer.user_scn, body_pos[:3], to_goal, radius=0.0005, rgba=(1, 0, 0, 0.5))
    
    # Add text overlay with simulation info
    text_to_display = (
        f"time: {data.time:.2f}s | "
        f"f_drive: {drive_freq} | "
        f"t_mag: {kp_mag:3g} | "
        f"avg. spd: {np.linalg.norm(data.qpos[:3] - initial_pos) / max(data.time, 1e-8):.2f} m/s | "
        f"vel: {np.linalg.norm(data.qvel[:3]):.2f} m/s"
    )
    add_text(data, viewer, text_to_display)


def _check_instability(model, data):
    """Check for simulation instability. Raises ValueError if unstable."""
    # Check for non-finite accelerations
    if not np.all(np.isfinite(data.qacc)):
        raise ValueError("Simulation unstable: Non-finite accelerations (qacc).")
    
    # Check if solver hit iteration limit
    if (data.solver_niter >= model.opt.iterations).any():
        raise ValueError("Simulation unstable: Solver iteration limit reached.")
    
    # Check for non-finite values in solver
    if not np.isfinite(data.solver_fwdinv[0]):
        raise ValueError("Simulation unstable: Non-finite values in solver.")


def _record_state(trajectory, data, step_cache=None):
    """Record current state to trajectory list."""
    entry = {
        "time": data.time,
        "pos": data.qpos[:3].copy(),
        "vel": data.qvel[:3].copy(),
        "quat": data.xquat[1].copy(),  # Main body quaternion
    }
    if step_cache is not None:
        # Store per-step torque decomposition for later power analysis
        if "tau_ext" in step_cache:
            entry["tau_ext"] = step_cache["tau_ext"].copy()
        if "tau_int" in step_cache:
            entry["tau_int"] = step_cache["tau_int"].copy()
        if "angle" in step_cache:
            entry["drive_angle"] = float(step_cache["angle"])
    trajectory.append(entry)


def _check_stuck_condition(data, last_check_pos, last_check_time, settle_time, 
                           stuck_check_interval, stuck_threshold, debug):
    """
    Check if robot is stuck. Returns updated (last_check_pos, last_check_time).
    Raises ValueError if stuck.
    """
    if data.time <= settle_time:
        return last_check_pos, last_check_time
    
    if last_check_pos is None:
        # Initialize check state right after settling is done
        return data.qpos[:2].copy(), data.time
    
    if data.time - last_check_time > stuck_check_interval:
        current_pos = data.qpos[:2]
        distance_moved = np.linalg.norm(current_pos - last_check_pos)
        
        if distance_moved < stuck_threshold:
            print(f"  [Debug] Stuck condition triggered: Moved {distance_moved:.6f}m < {stuck_threshold}m threshold in {stuck_check_interval}s.")
            if debug:
                print("\n--- SIMULATION STUCK ---")
                print(f"Time: {data.time:.4f}s")
                print(f"Position (qpos): {data.qpos[:7]}")  # Main body
                print(f"Velocity (qvel): {data.qvel[:6]}")  # Main body
                print("Applied forces on main body (xfrc_applied):")
                print(data.xfrc_applied[1])  # Main body is body 1
            raise ValueError("Simulation unstable: Robot is stuck.")
        
        return current_pos, data.time
    
    return last_check_pos, last_check_time


def _do_simulation_step(model, data, trajectory, kp_mag, drive_freq, settle_time,
                       mag_params,
                       last_check_pos, last_check_time, stuck_check_interval,
                       stuck_threshold, ignore_stuck_detection, debug,
                       benchmark=False, step_times=None):
    """
    Execute one simulation step: apply forces, step physics, check stability, record state.
    Returns (angle, updated_last_check_pos, updated_last_check_time).
    When benchmark=True, step_times is a dict (e.g. defaultdict(list)) to accumulate seconds per phase.
    """
    if benchmark and step_times is not None:
        t0 = time.perf_counter()
    step_cache = {}
    angle = _apply_magnetic_forces(model, data, kp_mag, drive_freq, settle_time, mag_params, step_cache)
    if benchmark and step_times is not None:
        step_times["apply_forces"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
    mujoco.mj_step(model, data)
    if benchmark and step_times is not None:
        step_times["mj_step"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
    _check_instability(model, data)
    _record_state(trajectory, data, step_cache)
    if benchmark and step_times is not None:
        step_times["record_state"].append(time.perf_counter() - t0)
    if not ignore_stuck_detection:
        last_check_pos, last_check_time = _check_stuck_condition(
            data, last_check_pos, last_check_time, settle_time,
            stuck_check_interval, stuck_threshold, debug
        )
    return angle, last_check_pos, last_check_time


def _maybe_capture_frame(renderer, cam, data, frames, next_frame_time, frame_time_step):
    """Capture a frame if it's time. Returns updated next_frame_time."""
    if renderer and data.time >= next_frame_time:
        renderer.update_scene(data, cam)
        pixels = renderer.render()
        frames.append(pixels)
        return next_frame_time + frame_time_step
    return next_frame_time


def _write_video(record_path, frames, framerate):
    """Write collected frames to video file."""
    if not frames:
        return
    
    print(f"Collected {len(frames)} frames. Writing video to {record_path}...")
    try:
        # Check for ffmpeg and download if missing
        try:
            imageio.plugins.ffmpeg.get_exe()
        except imageio.core.NeedDownloadError:
            print("\n--- FFMPEG dependency not found, attempting to download... ---")
            imageio.plugins.ffmpeg.download()
            print("--- FFMPEG downloaded successfully. ---")
        
        pathlib.Path(record_path).parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(record_path, fps=framerate) as writer:
            for frame in frames:
                writer.append_data(frame)
        print("Video writing complete.")
    except Exception as e:
        print(f"\n--- ERROR: Video writing failed unexpectedly ---")
        print(f"Could not write to {record_path}")
        print(f"Error: {e}")
        print("--------------------------------------------------\n")


def run_simulation(
    params, 
    mjcf_path="mulit_milli_quad/scene_4.xml", 
    sim_duration=10.0, 
    visualize=False, 
    record_path=None,
    benchmark=False,
    debug=False, 
    ignore_stuck_detection=False
    ):
    """
    Runs a MuJoCo simulation with given parameters and returns the trajectory.

    Args:
        params (dict): A dictionary of simulation parameters to tune.
            Example: {
                'ground_friction': [0.1, 0.005, 0.0001],
                'dof_damping': 3e-9,
                'solref': [0.004, 1],
                'solimp': [0.95, 0.99, 1e-3],
                'kp_mag': 2.5e-6
            }
        mjcf_path (str): Path to the MJCF XML file.
        sim_duration (float): The total duration of the simulation in seconds.
        visualize (bool): If True, launches the interactive viewer. This is ignored
            if `record_path` is set.
        record_path (str, optional): If provided, runs the simulation headlessly
            and records a video to this path.
        benchmark (bool): If True, run headlessly and print step-level timing
            (apply_forces, mj_step, record_state) after the run.
        debug (bool): If True, prints detailed information when a 'stuck'
            condition is detected before raising an error.
        ignore_stuck_detection (bool): If True, the simulation will not terminate
            early if the robot is detected as being stuck.

    Returns:
        list or None: A list of dictionaries representing the simulation
            trajectory, or None if the simulation was unstable.
    """
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    
    # Apply parameters from the params dict
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if ground_id != -1 and 'ground_friction' in params:
        model.geom_friction[ground_id] = params['ground_friction']
    
    if 'dof_damping' in params:
        model.dof_damping[-4:] = params['dof_damping']
        
    if 'solref' in params:
        model.opt.o_solref = params['solref']
        
    if 'solimp' in params:
        model.opt.o_solimp = params['solimp']

    kp_mag = params.get('kp_mag', 2.5e-6)
    drive_freq = params.get('drive_freq', 30)
    mag_params = params.get("mag_params", {})

    model.opt.timestep = SIM_TIMESTEP
    model.opt.enableflags |= 1 << 0  # enable override

    data = mujoco.MjData(model)
    _initialize_pose(data)
    initial_pos = data.qpos[:3].copy()
    
    trajectory = []
    frames = []
    renderer = None
    cam = None

    # If recording, set up an offline renderer and camera
    if record_path:
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = 1
        cam.distance = 0.2  # A bit further out for a good view

    try:
        # Initial step to settle the model (can be unstable at initialization)
        mujoco.mj_step(model, data)

        # --- Stuck condition detection state ---
        last_check_time = SETTLE_TIME
        last_check_pos = None  # Will be set after settle_time has passed

        # --- Main Loop ---
        if visualize and not record_path:
            # Interactive visualization mode
            paused = True
            def key_callback(keycode):
                nonlocal paused
                if chr(keycode) == ' ':
                    paused = not paused
            
            with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = 1
                viewer.cam.distance = 0.1
                
                while viewer.is_running() and data.time < sim_duration:
                    if not paused:
                        angle, last_check_pos, last_check_time = _do_simulation_step(
                            model, data, trajectory, kp_mag, drive_freq, SETTLE_TIME,
                            mag_params,
                            last_check_pos, last_check_time, STUCK_CHECK_INTERVAL,
                            STUCK_THRESHOLD, ignore_stuck_detection, debug
                        )
                        _update_viewer_overlays(viewer, data, drive_freq, kp_mag, initial_pos, angle)
                    
                    viewer.sync()
                    time.sleep(0.01)
        else:
            # Headless mode (with or without recording)
            frame_time_step = 1.0 / VIDEO_FRAMERATE
            next_frame_time = 0.0
            step_times = None
            if benchmark:
                from collections import defaultdict
                step_times = defaultdict(list)
            while data.time < sim_duration:
                angle, last_check_pos, last_check_time = _do_simulation_step(
                    model, data, trajectory, kp_mag, drive_freq, SETTLE_TIME,
                    mag_params,
                    last_check_pos, last_check_time, STUCK_CHECK_INTERVAL,
                    STUCK_THRESHOLD, ignore_stuck_detection, debug,
                    benchmark=benchmark, step_times=step_times
                )
                if record_path:
                    next_frame_time = _maybe_capture_frame(
                        renderer, cam, data, frames, next_frame_time, frame_time_step
                    )
            if benchmark and step_times:
                n = len(step_times["mj_step"])
                apply_s = sum(step_times["apply_forces"])
                step_s = sum(step_times["mj_step"])
                record_s = sum(step_times["record_state"])
                total_s = apply_s + step_s + record_s
                print(f"  Step timing ({n} steps): apply_forces={apply_s:.3f}s, mj_step={step_s:.3f}s, record_state={record_s:.3f}s (total={total_s:.3f}s)")
                
    except ValueError as e:
        if "Simulation unstable" in str(e) or "stuck in a loop" in str(e):
            # This can happen with unstable parameters. Return None to signal failure.
            print(f"  Simulation failed gracefully: {e}")
            return None
        else:
            # Re-raise any other unexpected ValueError
            raise e
            
    # --- Video Writing ---
    if record_path:
        _write_video(record_path, frames, VIDEO_FRAMERATE)

    # --- Cleanup and Return ---
    if renderer:
        renderer.close()

    # If the simulation produced NaN/Inf or is empty, it failed.
    if not trajectory or not np.all(np.isfinite([d['pos'][0] for d in trajectory])):
        print("Warning: Simulation produced NaN/Inf values or was empty. Penalizing.")
        return None
        
    return trajectory

if __name__ == "__main__":
    # Default parameters based on your last modifications
    default_params = {
        'ground_friction': [1e-5, 1e-5, 1e-5],
        'dof_damping': 7e-10,
        'solref': [0.004, 1],
        'solimp': [0.95, 0.99, 1e-3, 0.5, 1.0],
        'kp_mag': 2.5e-6,
        'mag_params': {'m_mag': MAGNETIC_MOMENT},
    }
    
    # Example of a headless run for optimization
    print("Running headless simulation to get steady-state velocity...")
    
    trajectory = run_simulation(default_params, sim_duration=5.0, visualize=False)

    if trajectory:
        # Calculate avg forward velocity from the trajectory
        start_state = trajectory[0]
        for state in trajectory:
            if state['time'] >= SETTLE_TIME:
                start_state = state
                break
        
        final_state = trajectory[-1]
        active_duration = final_state['time'] - start_state['time']
        
        avg_forward_velocity = 0
        if active_duration > 1e-6:
            forward_displacement = final_state['pos'][0] - start_state['pos'][0]
            avg_forward_velocity = forward_displacement / active_duration

        print(f"Steady-state velocity: {avg_forward_velocity:.4f} m/s")
    else:
        print("Simulation failed to produce a valid trajectory.")

    # Example of a visualized run to observe the behavior
    print("\nRunning simulation with visualization...")
    run_simulation(default_params, sim_duration=20.0, visualize=True, debug=True)
