import mujoco
import mujoco.viewer
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy
import copy
matplotlib.use('Agg')  # Use the non-interactive 'Agg' backend

mjcf_path = "mulit_milli_quad/scene_4.xml"

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

def add_text(data, viewer, input):
    # create an invisibale geom and add label on it
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_LABEL,
        size=np.array([0.2, 0.2, 0.2]),  # label_size
        pos=data.qpos[:3] + np.array([0.0, 0.0, 0.01]),  # label position, 1cm above
        mat=np.eye(3).flatten(),  # label orientation, no rotation
        rgba=np.array([0, 0, 0, 0])  # invisible
    )
    geom.label = input  # receive string input only
    viewer.user_scn.ngeom += 1

model = mujoco.MjModel.from_xml_path(mjcf_path)

# --- Friction Modification ---
# You can modify the friction properties of geoms to observe different behaviors.
# MuJoCo friction is defined by a 3-element array: [sliding, torsional, rolling].
# To make the robot slide more, you can decrease the 'sliding' friction component.
# Below is an example of how to change the friction of the ground plane.

# Print all geom names and their friction values to help identify them
print("Geoms and their friction values:")
for i in range(model.ngeom):
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    print(f"  ID: {i}, Name: {geom_name}, Friction: {model.geom_friction[i]}")

# Example: Modify ground friction
ground_geom_name = "floor"  # Change this name if your ground geom is named differently
ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, ground_geom_name)

if ground_id != -1:
    # Set new friction values here. For more sliding, reduce the first value.
    new_friction = [1e-5, 1e-5, 1e-5] 
    model.geom_friction[ground_id] = new_friction
    print(f"\nModified friction for '{ground_geom_name}' to {model.geom_friction[ground_id]}")
else:
    print(f"\nWarning: Geom '{ground_geom_name}' not found. Friction not modified.")
# -------------------------

model.opt.timestep = 1./2e3
model.dof_damping[-4:] = 7e-10
data = mujoco.MjData(model)
data.qpos[2] = 0.002  # Set initial position of the first joint
data.qacc[:] = 0  # Initialize accelerations to zero    

data.qpos[3:7] = [0, 0, 1, 0]  # Set initial position of the remaining joints
data.qpos[7:11] = np.pi*np.ones(4)
timestep = (model.opt.timestep
            * 1
            )
model.opt.enableflags |= 1 << 0  # enable override
# solreflimit="4e-3 1" solimplimit=".95 .99 1e-3"
# model.opt.iterations = 200
# model.opt.o_solref[0] = 4e-3
model.opt.o_solref[0] = 0.004
model.opt.o_solref[1] = 1
# model.opt.o_solref[1] = 100
model.opt.o_solimp[0] = 0.95 
model.opt.o_solimp[1] = 0.99 
model.opt.o_solimp[2] = 1e-3

pwm_freq = 1000
drive_freq = 30

# points_per_period = pwm_freq // drive_freq
# angles = np.linspace(0, 2 * np.pi, points_per_period, endpoint=False)

mujoco.mj_step(model, data)
data.qacc[:] = 0  # Initialize accelerations to zero    

rolls, pitches, yaws, jnts, angs = [], [], [], [], []

init_rs = []
for i in range(4):
    body_idx = i + 2
    body_quat = data.xquat[body_idx]
    body_frame = R.from_quat(body_quat)
    init_rs.append(deepcopy(body_frame))

paused = True

def key_callback(keycode):
    global paused
    if chr(keycode) == ' ':
        paused = not paused

viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
step_start = time.monotonic()
viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
viewer.cam.trackbodyid = 1
viewer.cam.distance = 0.1

# settling time before starting angle drive
settle_time = 0.1  # seconds

# Main simulation loop, with pause functionality
while viewer.is_running() and data.time < 50:

    if not paused:
        viewer.user_scn.ngeom = 0
        # Determine if we are before or after settle_time
        if data.time < settle_time:
            angle = 0.0
        else:
            angle = ((data.time - settle_time) * drive_freq * 2 * np.pi) % (2 * np.pi)
            # angle = angles[int(( (data.time - settle_time) * pwm_freq) % points_per_period)]
        angs.append(angle)
        print(f"time: {data.time:.5f}s, angle: {angle:.5f}")

        mag_norths = np.zeros((4, 3))
        goal_norths = np.zeros((4, 3))

        for i in range(4):
            body_idx = i + 2
            body_quat = data.xquat[body_idx]
            body_pos = data.xpos[body_idx]
            body_frame = R.from_quat(body_quat, scalar_first=True)
            arr_len = 0.01
            if i in [0,2]:
                body_frame_dir = np.array([1,0,0])
            else:
                body_frame_dir = np.array([-1,0,0])
            world_frame_dir = np.array([0,0,1])
            base_rot = init_rs[i]
            magnet_north = body_frame.as_matrix() @ body_frame_dir
            mag_norths[i] = magnet_north
            to = body_pos + arr_len * magnet_north

            rpy_rot = R.from_euler('y', angle, degrees=False)
            goal_north = rpy_rot.as_matrix() @ world_frame_dir
            goal_norths[i] = goal_north
            to_goal = body_pos + arr_len * goal_north

            add_visual_arrow(viewer.user_scn, body_pos[:3], to, rgba=(0, 1, 0, 1))
            add_visual_arrow(viewer.user_scn, body_pos[:3], to_goal, radius=0.0005, rgba=(1, 0, 0, 0.5))

            roll, pitch, yaw = body_frame.as_euler('zxy', degrees=False)
            kp_mag = 5e-6 * 0.5

            # Only apply magnetic forces after settle_time
            if data.time > settle_time:
                data.xfrc_applied[body_idx,3:] = (kp_mag) * np.cross(magnet_north, goal_north) * 1
                # data.ctrl[:] = -kp_mag*np.array([1, -1, 1, -1])

        add_text(data, viewer, 
                 f"""time: {data.time:.2f}s""" 
                 f"""   drive freq: {drive_freq}"""
                 f"""   mag torque: {kp_mag:3g}"""
                 f"""   body vel: {np.linalg.norm(data.qvel[:3]):.2f} m/s""")

        mujoco.mj_step(model, data)
    
    viewer.sync()

    time.sleep(0.01)
    
    time_until_next_step = timestep - (time.monotonic() - step_start)
    if time_until_next_step > 0:
        time.sleep(time_until_next_step)
    step_start = time.monotonic()

# plt.figure()
# tot_plots = 5
# plt.subplot(tot_plots, 1, 1)
# plt.plot(np.arange(len(rolls)) * timestep, rolls)
# plt.title('Roll')
# plt.subplot(tot_plots, 1, 2)
# plt.plot(np.arange(len(pitches)) * timestep, pitches)
# plt.title('Pitch')
# plt.subplot(tot_plots, 1, 3)
# plt.plot(np.arange(len(yaws)) * timestep, yaws)
# plt.title('Yaw')
# plt.subplot(tot_plots, 1, 4)
# plt.plot(np.arange(len(angs)) * timestep, angs)
# plt.title('Target Roll Angle')
# plt.subplot(tot_plots, 1, 5)
# plt.plot(np.arange(len(jnts)) * timestep, np.array(jnts) % (2 * np.pi))
# plt.title('Joint Position')
# plt.xlabel('Time (s)')
# plt.tight_layout()
# plt.savefig('orientation_plot.png') # Save the plot to a file
# print("Plot saved to orientation_plot.png")
