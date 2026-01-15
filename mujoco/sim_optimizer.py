import mujoco
import mujoco.viewer
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
import imageio
import imageio.plugins.ffmpeg
import pathlib


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


def run_simulation(params, mjcf_path="mulit_milli_quad/scene_4.xml", sim_duration=10.0, visualize=False, record_path=None, debug=False, ignore_stuck_detection=False):
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

    model.opt.timestep = 1./2e3
    model.opt.enableflags |= 1 << 0  # enable override

    data = mujoco.MjData(model)
    data.qpos[2] = 0.002
    data.qpos[3:7] = [0, 0, 1, 0]
    data.qpos[7:11] = np.pi * np.ones(4)
    
    initial_pos = data.qpos[:3].copy()

    init_rs = []
    drive_freq = params.get('drive_freq', 30)
    settle_time = 0.1
    velocities = []
    
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
        cam.distance = 0.2 # A bit further out for a good view

    try:
        # It's possible for the simulation to be unstable at initialization
        # depending on the parameters. We need to catch that here.
        mujoco.mj_step(model, data) # Initial step to settle the model

        for i in range(4):
            body_idx = i + 2
            body_quat = data.xquat[body_idx]
            body_frame = R.from_quat(body_quat)
            init_rs.append(deepcopy(body_frame))

        def simulation_step(viewer=None):
            """A single step of the simulation, with optional visualization."""
            if viewer:
                viewer.user_scn.ngeom = 0

            if data.time < settle_time:
                angle = 0.0
            else:
                angle = ((data.time - settle_time) * drive_freq * 2 * np.pi) % (2 * np.pi)
                # ang_range = np.pi/3
                # angle = ang_range * np.sin((data.time - settle_time) * drive_freq * 2 * np.pi) + ang_range/2    

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
                
                if data.time > settle_time:
                    data.xfrc_applied[body_idx, 3:] = kp_mag * np.cross(magnet_north, goal_north)

                if viewer:
                    arr_len = 0.01
                    to = body_pos + arr_len * magnet_north
                    to_goal = body_pos + arr_len * goal_north
                    add_visual_arrow(viewer.user_scn, body_pos[:3], to, rgba=(0, 1, 0, 1))
                    add_visual_arrow(viewer.user_scn, body_pos[:3], to_goal, radius=0.0005, rgba=(1, 0, 0, 0.5))

            if viewer:
                text_to_display = (
                    f"time: {data.time:.2f}s | "
                    f"f_drive: {drive_freq} | "
                    f"t_mag: {kp_mag:3g} | "
                    f"avg. spd: {np.linalg.norm(data.qpos[:3] - initial_pos) / max(data.time, 1e-8):.2f} m/s | "
                    f"vel: {np.linalg.norm(data.qvel[:3]):.2f} m/s"
                )
                add_text(data, viewer, text_to_display)

            mujoco.mj_step(model, data)

            # --- Robustness Check ---
            # These checks run only if mj_step completes successfully.
            # 1. Check for non-finite accelerations, which is what the MuJoCo warning flags.
            if not np.all(np.isfinite(data.qacc)):
                raise ValueError("Simulation unstable: Non-finite accelerations (qacc).")
            
            # 2. Check if the solver has struggled, which indicates instability.
            #    If any solver iteration count reaches the limit, the simulation is unstable.
            if (data.solver_niter >= model.opt.iterations).any():
                raise ValueError("Simulation unstable: Solver iteration limit reached.")

            # 3. Check for non-finite values in the forward dynamics computation.
            #    This is a definitive sign of simulation blow-up.
            if not np.isfinite(data.solver_fwdinv[0]):
                raise ValueError("Simulation unstable: Non-finite values in solver.")
            # --------------------------

            # Record state for trajectory
            trajectory.append({
                'time': data.time,
                'pos': data.qpos[:3].copy(),
                'vel': data.qvel[:3].copy(),
                'quat': data.xquat[1].copy() # Main body quaternion
            })

        # --- Stuck condition detection state ---
        stuck_check_interval = 5  # seconds
        stuck_threshold = 0.005     # meters over the interval
        last_check_time = settle_time
        last_check_pos = None # Will be set after settle_time has passed

        # --- Main Loop ---
        if visualize and not record_path:
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
                        simulation_step(viewer=viewer)

                        # --- Stuck Check ---
                        if not ignore_stuck_detection:
                            if data.time > settle_time:
                                if last_check_pos is None:
                                    # Initialize check state right after settling is done
                                    last_check_pos = data.qpos[:2].copy()
                                    last_check_time = data.time
                                
                                if data.time - last_check_time > stuck_check_interval:
                                    current_pos = data.qpos[:2]
                                    distance_moved = np.linalg.norm(current_pos - last_check_pos)
                                    if distance_moved < stuck_threshold:
                                        print(f"  [Debug] Stuck condition triggered: Moved {distance_moved:.6f}m < {stuck_threshold}m threshold in {stuck_check_interval}s.")
                                        if debug:
                                            print("\n--- SIMULATION STUCK ---")
                                            print(f"Time: {data.time:.4f}s")
                                            print(f"Position (qpos): {data.qpos[:7]}") # Main body
                                            print(f"Velocity (qvel): {data.qvel[:6]}") # Main body
                                            print("Applied forces on main body (xfrc_applied):")
                                            print(data.xfrc_applied[1]) # Main body is body 1
                                        raise ValueError("Simulation unstable: Robot is stuck.")
                                    last_check_time = data.time
                                    last_check_pos = current_pos
                        # -------------------

                    viewer.sync()
                    time.sleep(0.01)
        else:  # Headless run (with or without recording)
            # --- Performance improvement for recording ---
            framerate = 60.0
            frame_time_step = 1.0 / framerate
            next_frame_time = 0.0
            # ---------------------------------------------
            
            while data.time < sim_duration:
                simulation_step()

                # --- Stuck Check ---
                if not ignore_stuck_detection:
                    if data.time > settle_time:
                        if last_check_pos is None:
                            # Initialize check state right after settling is done
                            last_check_pos = data.qpos[:2].copy()
                            last_check_time = data.time
                        
                        if data.time - last_check_time > stuck_check_interval:
                            current_pos = data.qpos[:2]
                            distance_moved = np.linalg.norm(current_pos - last_check_pos)
                            if distance_moved < stuck_threshold:
                                print(f"  [Debug] Stuck condition triggered: Moved {distance_moved:.6f}m < {stuck_threshold}m threshold in {stuck_check_interval}s.")
                                if debug:
                                    print("\n--- SIMULATION STUCK ---")
                                    print(f"Time: {data.time:.4f}s")
                                    print(f"Position (qpos): {data.qpos[:7]}") # Main body
                                    print(f"Velocity (qvel): {data.qvel[:6]}") # Main body
                                    print("Applied forces on main body (xfrc_applied):")
                                    print(data.xfrc_applied[1]) # Main body is body 1
                                raise ValueError("Simulation unstable: Robot is stuck.")
                            last_check_time = data.time
                            last_check_pos = current_pos
                # -------------------

                if renderer and data.time >= next_frame_time:
                    renderer.update_scene(data, cam)
                    pixels = renderer.render()
                    frames.append(pixels)
                    next_frame_time += frame_time_step
                
    except ValueError as e:
        if "Simulation unstable" in str(e) or "stuck in a loop" in str(e):
            # This can happen with unstable parameters. Return None to signal failure.
            print(f"  Simulation failed gracefully: {e}")
            return None
        else:
            # Re-raise any other unexpected ValueError
            raise e
            
    # --- Video Writing ---
    if record_path and frames:
        print(f"Collected {len(frames)} frames. Writing video to {record_path}...")
        try:
            # Proactively check for ffmpeg and offer to download if missing.
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
        'kp_mag': 2.5e-6
    }
    
    # Example of a headless run for optimization
    print("Running headless simulation to get steady-state velocity...")
    
    trajectory = run_simulation(default_params, sim_duration=5.0, visualize=False)

    if trajectory:
        # Calculate avg forward velocity from the trajectory
        settle_time = 0.1
        start_state = trajectory[0]
        for state in trajectory:
            if state['time'] >= settle_time:
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
