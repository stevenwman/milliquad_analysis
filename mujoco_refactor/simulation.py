"""
Core MuJoCo simulation engine for the LEGO milliquad robot.

Computes two types of magnetic torques:
1. External torques: from a rotating drive field aligning magnets to a goal direction.
2. Inter-joint torques: dipole-dipole magnetic coupling between legs (τ = m × B).

Usage:
    from simulation import run_simulation
    trajectory = run_simulation(params, sim_duration=5.0)
"""

import time
from typing import Any
import os

import imageio
import imageio.plugins.ffmpeg
import mujoco
import mujoco.viewer
# Suppress MuJoCo's C-level "Nan, Inf or huge value" stderr warnings —
# _check_instability already catches these conditions programmatically.
mujoco.set_mju_user_warning(lambda msg: None)
import numpy as np
import pathlib
from scipy.spatial.transform import Rotation as R

from config import (
    CAMERA_DISTANCE_RECORD,
    CAMERA_DISTANCE_VIEWER,
    INITIAL_LEG_ANGLES,
    INITIAL_QUATERNION,
    INITIAL_Z_HEIGHT,
    LEG_BODY_OFFSET,
    MAGNETIC_FIELD_MAGNITUDE,
    MAGNETIC_MOMENT,
    MU0_OVER_4PI,
    PACKAGE_DIR,
    R_EPS,
    SETTLE_TIME,
    SIM_TIMESTEP,
    STUCK_CHECK_INTERVAL,
    STUCK_THRESHOLD,
    VIDEO_FRAMERATE,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def add_visual_arrow(scene, from_point, to_point, radius=0.001, rgba=(0, 0, 1, 1)):
    """Adds a single visual arrow to the mjvScene (visual-only, no physics)."""
    if scene.ngeom >= scene.maxgeom:
        print("Warning: Maximum number of geoms reached. Cannot add arrow.")
        return

    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.array([radius, radius, np.linalg.norm(to_point - from_point)]),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, radius, from_point, to_point)
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
        rgba=np.array([0, 0, 0, 0]),
    )
    geom.label = text_input
    viewer.user_scn.ngeom += 1


# ---------------------------------------------------------------------------
# Simulation helper functions
# ---------------------------------------------------------------------------

def _initialize_pose(
    data,
    init_yaw_jitter_deg: float = 0.0,
    rng: np.random.Generator | None = None,
) -> None:
    """Initialize the robot pose. Main body raised, rotated 180° about z, legs at π."""
    data.qpos[2] = INITIAL_Z_HEIGHT
    if init_yaw_jitter_deg > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        yaw = np.deg2rad(rng.uniform(-init_yaw_jitter_deg, init_yaw_jitter_deg))
        base = R.from_quat(INITIAL_QUATERNION, scalar_first=True)
        jitter = R.from_euler("z", yaw, degrees=False)
        data.qpos[3:7] = (jitter * base).as_quat(scalar_first=True)
    else:
        data.qpos[3:7] = INITIAL_QUATERNION
    data.qpos[7:11] = INITIAL_LEG_ANGLES * np.ones(4)


def _compute_drive_angle(sim_time: float, drive_freq: float, settle_time: float) -> float:
    """Compute the drive angle based on time. Returns 0 during settle period."""
    if sim_time < settle_time:
        return 0.0
    return ((sim_time - settle_time) * drive_freq * 2 * np.pi) % (2 * np.pi)


def _get_magnet_state(data, i: int) -> tuple[int, np.ndarray, np.ndarray]:
    """Return (body_idx, pos_world, north_world_unit) for leg i (0-3)."""
    body_idx = i + LEG_BODY_OFFSET
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


def _dipole_field(mj: np.ndarray, r_vec: np.ndarray) -> np.ndarray:
    """Magnetic field B at offset r_vec from dipole moment mj."""
    r = max(np.linalg.norm(r_vec), R_EPS)
    rhat = r_vec / r
    return MU0_OVER_4PI * (1.0 / r**3) * (3.0 * np.dot(mj, rhat) * rhat - mj)


def _compute_external_torques(data, angle: float, kp_mag: float, settle_time: float) -> np.ndarray:
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


def _compute_interjoint_torques(data, m_mag: float) -> np.ndarray:
    """Return tau_int[4,3] world torques from dipole-dipole coupling (τ = m × B)."""
    tau_int = np.zeros((4, 3))
    if m_mag == 0.0:
        return tau_int

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
        tau_int[i] = np.cross(m[i], Bi)

    return tau_int


def _apply_magnetic_forces(
    data,
    kp_mag: float,
    drive_freq: float,
    settle_time: float,
    mag_params: dict,
    step_cache: dict,
) -> float:
    """
    Apply magnetic torques to 4 leg bodies based on external drive and inter-joint coupling.

    Side effects:
        - Writes data.xfrc_applied[:, 3:6] with total world torques.
        - Stores tau_ext, tau_int, angle into step_cache for later use.
    """
    angle = _compute_drive_angle(data.time, drive_freq, settle_time)

    data.xfrc_applied[:, :] = 0.0

    tau_ext = _compute_external_torques(data, angle, kp_mag, settle_time)

    if "m_mag" not in mag_params:
        raise ValueError(
            "mag_params missing required key 'm_mag'. "
            "Define it in the optimization loop / caller."
        )
    tau_int = _compute_interjoint_torques(data, m_mag=mag_params["m_mag"])

    for i in range(4):
        body_idx, _, _ = _get_magnet_state(data, i)
        data.xfrc_applied[body_idx, 3:6] += tau_ext[i] + tau_int[i]

    # Angular velocity of each leg body in world frame (for power computation).
    # Captured *before* mj_step so omega is at the same instant as the torques.
    omega = np.zeros((4, 3))
    for i in range(4):
        omega[i] = data.cvel[i + LEG_BODY_OFFSET, :3]

    step_cache["tau_ext"] = tau_ext
    step_cache["tau_int"] = tau_int
    step_cache["omega"] = omega
    step_cache["angle"] = angle

    return angle


def _update_viewer_overlays(viewer, data, drive_freq, kp_mag, initial_pos, angle):
    """Update visual overlays (arrows and text) in the viewer."""
    viewer.user_scn.ngeom = 0

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

    text_to_display = (
        f"time: {data.time:.2f}s | "
        f"f_drive: {drive_freq} | "
        f"t_mag: {kp_mag:3g} | "
        f"avg. spd: {np.linalg.norm(data.qpos[:3] - initial_pos) / max(data.time, 1e-8):.2f} m/s | "
        f"vel: {np.linalg.norm(data.qvel[:3]):.2f} m/s"
    )
    add_text(data, viewer, text_to_display)


def _check_instability(model, data) -> None:
    """Check for simulation instability. Raises ValueError if unstable."""
    if not np.all(np.isfinite(data.qacc)):
        raise ValueError("Simulation unstable: Non-finite accelerations (qacc).")

    if (data.solver_niter >= model.opt.iterations).any():
        raise ValueError("Simulation unstable: Solver iteration limit reached.")

    if not np.isfinite(data.solver_fwdinv[0]):
        raise ValueError("Simulation unstable: Non-finite values in solver.")

    # Check MuJoCo's internal warning flags (set when it prints "Nan, Inf or huge value")
    if data.warning.number.any():
        active = [
            mujoco.mjtWarning(i).name
            for i in range(data.warning.number.shape[0])
            if data.warning.number[i] > 0
        ]
        raise ValueError(f"Simulation unstable: MuJoCo warnings triggered: {active}")


def _record_state(trajectory: list[dict], data, step_cache: dict | None = None) -> None:
    """Record current state to trajectory list."""
    entry = {
        "time": data.time,
        "pos": data.qpos[:3].copy(),
        "vel": data.qvel[:3].copy(),
        "quat": data.xquat[1].copy(),
    }
    if step_cache is not None:
        if "tau_ext" in step_cache:
            entry["tau_ext"] = step_cache["tau_ext"].copy()
        if "tau_int" in step_cache:
            entry["tau_int"] = step_cache["tau_int"].copy()
        if "omega" in step_cache:
            entry["omega"] = step_cache["omega"].copy()
        if "angle" in step_cache:
            entry["drive_angle"] = float(step_cache["angle"])
    trajectory.append(entry)


def _check_stuck_condition(
    data,
    last_check_pos: np.ndarray | None,
    last_check_time: float,
    settle_time: float,
    stuck_check_interval: float,
    stuck_threshold: float,
    debug: bool,
) -> tuple[np.ndarray | None, float]:
    """
    Check if robot is stuck. Returns updated (last_check_pos, last_check_time).
    Raises ValueError if stuck.
    """
    if data.time <= settle_time:
        return last_check_pos, last_check_time

    if last_check_pos is None:
        return data.qpos[:2].copy(), data.time

    if data.time - last_check_time > stuck_check_interval:
        current_pos = data.qpos[:2]
        distance_moved = np.linalg.norm(current_pos - last_check_pos)

        if distance_moved < stuck_threshold:
            print(f"  [Debug] Stuck condition triggered: Moved {distance_moved:.6f}m < {stuck_threshold}m threshold in {stuck_check_interval}s.")
            if debug:
                print("\n--- SIMULATION STUCK ---")
                print(f"Time: {data.time:.4f}s")
                print(f"Position (qpos): {data.qpos[:7]}")
                print(f"Velocity (qvel): {data.qvel[:6]}")
                print("Applied forces on main body (xfrc_applied):")
                print(data.xfrc_applied[1])
            raise ValueError("Simulation unstable: Robot is stuck.")

        return current_pos, data.time

    return last_check_pos, last_check_time


def _do_simulation_step(
    model,
    data,
    trajectory: list[dict],
    kp_mag: float,
    drive_freq: float,
    settle_time: float,
    mag_params: dict,
    last_check_pos: np.ndarray | None,
    last_check_time: float,
    stuck_check_interval: float,
    stuck_threshold: float,
    ignore_stuck_detection: bool,
    debug: bool,
    benchmark: bool = False,
    step_times: dict | None = None,
) -> tuple[float, np.ndarray | None, float]:
    """
    Execute one simulation step: apply forces, step physics, check stability, record state.
    Returns (angle, updated_last_check_pos, updated_last_check_time).
    """
    if benchmark and step_times is not None:
        t0 = time.perf_counter()
    step_cache = {}
    angle = _apply_magnetic_forces(data, kp_mag, drive_freq, settle_time, mag_params, step_cache)
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


def _maybe_capture_frame(
    renderer, cam, data, frames: list, next_frame_time: float, frame_time_step: float
) -> float:
    """Capture a frame if it's time. Returns updated next_frame_time."""
    if renderer and data.time >= next_frame_time:
        renderer.update_scene(data, cam)
        pixels = renderer.render()
        frames.append(pixels)
        return next_frame_time + frame_time_step
    return next_frame_time


def _write_video(record_path: str, frames: list, framerate: float) -> None:
    """Write collected frames to video file."""
    if not frames:
        return

    print(f"Collected {len(frames)} frames. Writing video to {record_path}...")
    try:
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_simulation(
    params: dict[str, Any],
    mjcf_path: str = str(PACKAGE_DIR / "multi_milli_quad" / "scene_4.xml"),
    sim_duration: float = 10.0,
    visualize: bool = False,
    record_path: str | None = None,
    benchmark: bool = False,
    debug: bool = False,
    ignore_stuck_detection: bool = False,
    progress: bool = False,
    wall_timeout: float | None = None,
    init_yaw_jitter_deg: float = 0.0,
    rng_seed: int | None = None,
) -> list[dict] | None:
    """
    Run a MuJoCo simulation with given parameters and return the trajectory.

    Args:
        params: Simulation parameters dict. Expected keys:
            ground_friction, dof_damping, solref, solimp, kp_mag, mag_params
        mjcf_path: Path to the MJCF XML file.
        sim_duration: Total simulation time in seconds.
        visualize: Launch interactive viewer (ignored if record_path is set).
        record_path: If provided, run headless and record video here.
        benchmark: Print step-level timing after the run.
        debug: Print detailed info on stuck detection.
        ignore_stuck_detection: Skip early termination for stuck robots.
        progress: Print 20% timestep milestones during headless runs.
        wall_timeout: Optional wall-clock timeout in seconds. If exceeded, abort.

    Returns:
        List of trajectory dicts, or None if simulation was unstable.
    """
    model = mujoco.MjModel.from_xml_path(mjcf_path)

    # Apply parameters (all required — caller must provide a fully-populated dict)
    model.dof_damping[-4:] = params['dof_damping']
    model.opt.o_solref = params['solref']
    model.opt.o_solimp = params['solimp']
    # o_friction is [tangent1, tangent2, spin, rolling1, rolling2];
    # params['ground_friction'] is [sliding, torsional, rolling].
    gf = params['ground_friction']
    model.opt.o_friction[:] = [gf[0], gf[0], gf[1], gf[2], gf[2]]

    kp_mag = params['kp_mag']
    drive_freq = params['drive_freq']
    mag_params = params['mag_params']

    model.opt.timestep = SIM_TIMESTEP
    # Enable global contact parameter overrides (o_solref, o_solimp, o_friction)
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_OVERRIDE
    # condim=6: enable torsional + rolling friction (default condim=3 ignores them)
    model.geom_condim[:] = 6

    data = mujoco.MjData(model)
    rng = np.random.default_rng(rng_seed) if rng_seed is not None else None
    _initialize_pose(data, init_yaw_jitter_deg=init_yaw_jitter_deg, rng=rng)
    initial_pos = data.qpos[:3].copy()

    trajectory = []
    frames = []
    renderer = None
    cam = None

    if record_path:
        renderer = mujoco.Renderer(model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = 1
        cam.distance = CAMERA_DISTANCE_RECORD

    progress_thresholds = None
    next_progress_idx = 0
    t_wall_start = time.perf_counter()
    if progress and sim_duration > 0:
        progress_thresholds = [0.2, 0.4, 0.6, 0.8, 1.0]
        scene_name = pathlib.Path(mjcf_path).stem
        seed_label = rng_seed if rng_seed is not None else "none"
        print(f"[sim pid={os.getpid()} scene={scene_name} seed={seed_label}] start")
    try:
        mujoco.mj_step(model, data)

        last_check_time = SETTLE_TIME
        last_check_pos = None

        if visualize and not record_path:
            paused = True

            def key_callback(keycode):
                nonlocal paused
                if chr(keycode) == ' ':
                    paused = not paused

            with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = 1
                viewer.cam.distance = CAMERA_DISTANCE_VIEWER

                while viewer.is_running() and data.time < sim_duration:
                    if wall_timeout is not None and (time.perf_counter() - t_wall_start) > wall_timeout:
                        raise ValueError(
                            f"Simulation unstable: Worker wall-time timeout exceeded ({wall_timeout:.1f}s)."
                        )
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
            frame_time_step = 1.0 / VIDEO_FRAMERATE
            next_frame_time = 0.0
            step_times = None
            if benchmark:
                from collections import defaultdict
                step_times = defaultdict(list)

            while data.time < sim_duration:
                if wall_timeout is not None and (time.perf_counter() - t_wall_start) > wall_timeout:
                    raise ValueError(
                        f"Simulation unstable: Worker wall-time timeout exceeded ({wall_timeout:.1f}s)."
                    )
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
                if progress_thresholds is not None:
                    frac = min(1.0, data.time / sim_duration)
                    while (
                        next_progress_idx < len(progress_thresholds)
                        and frac >= progress_thresholds[next_progress_idx]
                    ):
                        pct = int(progress_thresholds[next_progress_idx] * 100)
                        print(
                            f"[sim pid={os.getpid()} scene={scene_name} seed={seed_label}] {pct}% "
                            f"(t={data.time:.2f}/{sim_duration:.2f}s)"
                        )
                        next_progress_idx += 1

            if benchmark and step_times:
                n = len(step_times["mj_step"])
                apply_s = sum(step_times["apply_forces"])
                step_s = sum(step_times["mj_step"])
                record_s = sum(step_times["record_state"])
                total_s = apply_s + step_s + record_s
                print(f"  Step timing ({n} steps): apply_forces={apply_s:.3f}s, mj_step={step_s:.3f}s, record_state={record_s:.3f}s (total={total_s:.3f}s)")

    except ValueError as e:
        if "Simulation unstable" in str(e) or "stuck in a loop" in str(e):
            print(f"  Simulation failed gracefully: {e}")
            return None
        else:
            raise

    if record_path:
        _write_video(record_path, frames, VIDEO_FRAMERATE)

    if renderer:
        renderer.close()

    if not trajectory or not np.all(np.isfinite([d['pos'][0] for d in trajectory])):
        print("Warning: Simulation produced NaN/Inf values or was empty. Penalizing.")
        return None

    return trajectory


if __name__ == "__main__":
    from config import DEFAULT_CTRL_FREQ
    default_params = {
        'ground_friction': [1e-5, 1e-5, 1e-5],
        'dof_damping': 7e-10,
        'solref': [0.004, 1],
        'solimp': [0.95, 0.99, 1e-3, 0.5, 1.0],
        'kp_mag': 2.5e-6,
        'drive_freq': DEFAULT_CTRL_FREQ,
        'mag_params': {'m_mag': MAGNETIC_MOMENT},
    }

    print("Running headless simulation to get steady-state velocity...")

    trajectory = run_simulation(default_params, sim_duration=5.0, visualize=False)

    if trajectory:
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

    print("\nRunning simulation with visualization...")
    run_simulation(default_params, sim_duration=20.0, visualize=True, debug=True)
