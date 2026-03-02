"""
Core MuJoCo simulation engine for the LEGO milliquad robot (unified, 16-param).

Supports both yaw jitter (flat/step) and spawn offset (rough) initialization.
All scene XMLs have RK4 integrator baked in.

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
    spawn_offset: tuple[float, float, float] | None = None,
) -> None:
    """Initialize the robot pose.

    Args:
        init_yaw_jitter_deg: Random yaw perturbation (flat/step terrain).
        rng: RNG for yaw jitter.
        spawn_offset: (dx, dy, dz) position offset (rough terrain).
    """
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

    if spawn_offset is not None:
        data.qpos[0] += spawn_offset[0]
        data.qpos[1] += spawn_offset[1]
        data.qpos[2] += spawn_offset[2]


def _compute_drive_angle(sim_time: float, drive_freq: float, settle_time: float) -> float:
    """Compute the drive angle based on time. Returns 0 during settle period."""
    if sim_time < settle_time:
        return 0.0
    return ((sim_time - settle_time) * drive_freq * 2 * np.pi) % (2 * np.pi)


def _quat_rotate_vec(q_wxyz, v):
    """Rotate vector v by quaternion q in (w,x,y,z) format. Pure numpy."""
    w = q_wxyz[0]
    q_xyz = q_wxyz[1:4]
    t = 2.0 * np.cross(q_xyz, v)
    return v + w * t + np.cross(q_xyz, t)


def _quat_rotate_vec_batch(q_wxyz, v):
    """Batched quaternion-vector rotation. q_wxyz: (N,4), v: (N,3) -> (N,3)."""
    w = q_wxyz[:, 0:1]
    q_xyz = q_wxyz[:, 1:4]
    t = 2.0 * np.cross(q_xyz, v)
    return v + w * t + np.cross(q_xyz, t)


# Pre-allocated body direction vectors
_BODY_DIR_POS_X = np.array([1.0, 0.0, 0.0])
_BODY_DIR_NEG_X = np.array([-1.0, 0.0, 0.0])
_BODY_DIRS = np.array([[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=float)
_LEG_BODY_SLICE = slice(LEG_BODY_OFFSET, LEG_BODY_OFFSET + 4)


def _get_all_magnet_states(data):
    """Compute positions and north directions for all 4 legs at once."""
    pos = data.xpos[_LEG_BODY_SLICE]
    quats = data.xquat[_LEG_BODY_SLICE]
    north = _quat_rotate_vec_batch(quats, _BODY_DIRS)
    norms = np.linalg.norm(north, axis=1, keepdims=True)
    np.maximum(norms, 1e-16, out=norms)
    north /= norms
    return pos, north


def _compute_external_torques(data, angle: float, kp_mag: float, settle_time: float,
                               north: np.ndarray) -> np.ndarray:
    """Return tau_ext[4,3] world torques from external drive."""
    if data.time <= settle_time:
        return np.zeros((4, 3))

    goal = np.array([np.sin(angle), 0.0, np.cos(angle)])
    return kp_mag * np.cross(north, goal)


_DIAG_IDX = np.arange(4)


def _compute_interjoint_torques(m_mag: float, pos: np.ndarray,
                                north: np.ndarray) -> np.ndarray:
    """Return tau_int[4,3] world torques from dipole-dipole coupling."""
    if m_mag == 0.0:
        return np.zeros((4, 3))

    m = m_mag * north
    r_vecs = pos[:, None, :] - pos[None, :, :]
    r_norms = np.linalg.norm(r_vecs, axis=-1)
    np.maximum(r_norms, R_EPS, out=r_norms)
    r_hat = r_vecs / r_norms[..., None]
    inv_r3 = 1.0 / (r_norms ** 3)
    m_dot_rhat = np.einsum('jk,ijk->ij', m, r_hat)
    B_ij = MU0_OVER_4PI * inv_r3[..., None] * (
        3.0 * m_dot_rhat[..., None] * r_hat - m[None, :, :]
    )
    B_ij[_DIAG_IDX, _DIAG_IDX, :] = 0.0
    B_total = B_ij.sum(axis=1)
    return np.cross(m, B_total)


def _apply_magnetic_forces(
    data,
    kp_mag: float,
    drive_freq: float,
    settle_time: float,
    mag_params: dict,
    step_cache: dict,
) -> float:
    """Apply magnetic torques to 4 leg bodies."""
    angle = _compute_drive_angle(data.time, drive_freq, settle_time)
    data.xfrc_applied[:, :] = 0.0
    pos, north = _get_all_magnet_states(data)
    tau_ext = _compute_external_torques(data, angle, kp_mag, settle_time, north)

    if "m_mag" not in mag_params:
        raise ValueError("mag_params missing required key 'm_mag'.")
    tau_int = _compute_interjoint_torques(m_mag=mag_params["m_mag"], pos=pos, north=north)

    data.xfrc_applied[_LEG_BODY_SLICE, 3:6] += tau_ext + tau_int

    omega = data.cvel[_LEG_BODY_SLICE, :3].copy()

    step_cache["tau_ext"] = tau_ext
    step_cache["tau_int"] = tau_int
    step_cache["omega"] = omega
    step_cache["angle"] = angle
    step_cache["north"] = north

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

    if data.warning.number.any():
        active = [
            mujoco.mjtWarning(i).name
            for i in range(data.warning.number.shape[0])
            if data.warning.number[i] > 0
        ]
        raise ValueError(f"Simulation unstable: MuJoCo warnings triggered: {active}")


def _extract_contact_data(model, data) -> dict:
    """Extract per-leg and chassis contact summary from MuJoCo contact state.

    Returns dict with:
        leg_in_contact:      (4,) bool
        leg_normal_force:    (4,) float  — total normal force per leg (N)
        leg_tangent_force:   (4,) float  — total tangential force magnitude per leg (N)
        leg_contact_pos:     (4,3) float — centroid of contact positions per leg (m)
        body_in_contact:     bool        — chassis touching terrain?
        body_normal_force:   float       — total normal force on chassis (N)
        body_tangent_force:  float       — total tangential force on chassis (N)
        total_ncon:          int         — total active contacts
    """
    _CHASSIS_BODY = LEG_BODY_OFFSET - 1  # body 1

    leg_in_contact = np.zeros(4, dtype=bool)
    leg_normal_force = np.zeros(4)
    leg_tangent_force = np.zeros(4)
    leg_contact_pos_sum = np.zeros((4, 3))
    leg_contact_count = np.zeros(4, dtype=int)

    body_in_contact = False
    body_normal_force = 0.0
    body_tangent_force = 0.0

    force_buf = np.zeros(6)  # reusable buffer for mj_contactForce

    for i in range(data.ncon):
        contact = data.contact[i]
        body1 = model.geom_bodyid[contact.geom1]
        body2 = model.geom_bodyid[contact.geom2]

        # Chassis-terrain contact: body 1 vs world (0)
        if (body1 == _CHASSIS_BODY and body2 == 0) or (body2 == _CHASSIS_BODY and body1 == 0):
            mujoco.mj_contactForce(model, data, i, force_buf)
            body_in_contact = True
            body_normal_force += force_buf[0]
            body_tangent_force += np.sqrt(force_buf[1] ** 2 + force_buf[2] ** 2)
            continue

        # Leg-terrain contact: one body is a leg (2-5), other is world (0)
        leg_idx = -1
        if LEG_BODY_OFFSET <= body1 < LEG_BODY_OFFSET + 4 and body2 == 0:
            leg_idx = body1 - LEG_BODY_OFFSET
        elif LEG_BODY_OFFSET <= body2 < LEG_BODY_OFFSET + 4 and body1 == 0:
            leg_idx = body2 - LEG_BODY_OFFSET
        if leg_idx < 0:
            continue

        mujoco.mj_contactForce(model, data, i, force_buf)
        leg_in_contact[leg_idx] = True
        leg_normal_force[leg_idx] += force_buf[0]
        leg_tangent_force[leg_idx] += np.sqrt(force_buf[1] ** 2 + force_buf[2] ** 2)
        leg_contact_pos_sum[leg_idx] += contact.pos
        leg_contact_count[leg_idx] += 1

    # Compute centroid for legs with contacts
    mask = leg_contact_count > 0
    leg_contact_pos = np.zeros((4, 3))
    if mask.any():
        leg_contact_pos[mask] = leg_contact_pos_sum[mask] / leg_contact_count[mask, None]

    return {
        "leg_in_contact": leg_in_contact,
        "leg_normal_force": leg_normal_force,
        "leg_tangent_force": leg_tangent_force,
        "leg_contact_pos": leg_contact_pos,
        "body_in_contact": body_in_contact,
        "body_normal_force": body_normal_force,
        "body_tangent_force": body_tangent_force,
        "total_ncon": int(data.ncon),
    }


def _record_state(trajectory: list[dict], model, data, step_cache: dict | None = None) -> None:
    """Record current state to trajectory list."""
    entry = {
        "time": data.time,
        "pos": data.qpos[:3].copy(),
        "vel": data.qvel[:3].copy(),
        "quat": data.xquat[1].copy(),
        "joint_pos": data.qpos[7:11].copy(),
        "joint_vel": data.qvel[6:10].copy(),
        "leg_xquat": data.xquat[_LEG_BODY_SLICE].copy(),
        "leg_xpos": data.xpos[_LEG_BODY_SLICE].copy(),
    }
    # Contact data
    entry.update(_extract_contact_data(model, data))

    if step_cache is not None:
        if "tau_ext" in step_cache:
            entry["tau_ext"] = step_cache["tau_ext"].copy()
        if "tau_int" in step_cache:
            entry["tau_int"] = step_cache["tau_int"].copy()
        if "omega" in step_cache:
            entry["omega"] = step_cache["omega"].copy()
        if "angle" in step_cache:
            entry["drive_angle"] = float(step_cache["angle"])
        if "north" in step_cache:
            entry["north"] = step_cache["north"].copy()
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
    """Check if robot is stuck. Raises ValueError if stuck."""
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
    """Execute one simulation step."""
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
    _record_state(trajectory, model, data, step_cache)
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
    mjcf_path: str = str(PACKAGE_DIR / "robots" / "quad" / "scene_4_flat.xml"),
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
    spawn_offset: tuple[float, float, float] | None = None,
) -> list[dict] | None:
    """
    Run a MuJoCo simulation with given parameters and return the trajectory.

    Args:
        params: Simulation parameters dict. Expected keys:
            ground_friction, dof_damping, solref, solimp, kp_mag, mag_params
            noslip_iterations, noslip_tolerance, margin
        mjcf_path: Path to the MJCF XML file.
        sim_duration: Total simulation time in seconds.
        visualize: Launch interactive viewer (ignored if record_path is set).
        record_path: If provided, run headless and record video here.
        benchmark: Print step-level timing after the run.
        debug: Print detailed info on stuck detection.
        ignore_stuck_detection: Skip early termination for stuck robots.
        progress: Print 20% timestep milestones during headless runs.
        wall_timeout: Optional wall-clock timeout in seconds.
        init_yaw_jitter_deg: Random yaw perturbation in degrees (flat/step).
        rng_seed: Seed for yaw jitter RNG.
        spawn_offset: (dx, dy, dz) position offset (rough terrain).

    Returns:
        List of trajectory dicts, or None if simulation was unstable.
    """
    model = mujoco.MjModel.from_xml_path(mjcf_path)

    # Apply parameters
    model.dof_damping[-4:] = params['dof_damping']
    model.opt.o_solref = params['solref']
    model.opt.o_solimp = params['solimp']
    gf = params['ground_friction']
    model.opt.o_friction[:] = [gf[0], gf[0], gf[1], gf[2], gf[2]]

    if 'noslip_iterations' in params:
        model.opt.noslip_iterations = int(params['noslip_iterations'])
    if 'noslip_tolerance' in params:
        model.opt.noslip_tolerance = float(params['noslip_tolerance'])
    if 'margin' in params:
        model.opt.o_margin = float(params['margin'])

    kp_mag = params['kp_mag']
    drive_freq = params['drive_freq']
    mag_params = params['mag_params']

    model.opt.timestep = SIM_TIMESTEP
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_OVERRIDE
    model.geom_condim[:] = 6

    data = mujoco.MjData(model)
    rng = np.random.default_rng(rng_seed) if rng_seed is not None else None
    _initialize_pose(
        data,
        init_yaw_jitter_deg=init_yaw_jitter_deg,
        rng=rng,
        spawn_offset=spawn_offset,
    )
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
            return None
        else:
            raise

    if record_path:
        _write_video(record_path, frames, VIDEO_FRAMERATE)

    if renderer:
        renderer.close()

    if not trajectory or not np.all(np.isfinite([d['pos'][0] for d in trajectory])):
        return None

    return trajectory
