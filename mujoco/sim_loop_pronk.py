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

mjcf_path = "one_milli_quad/scene.xml"


def add_visual_arrow(scene, from_point, to_point, radius=0.001, rgba=(0, 0, 1, 1)):
    """
    Adds a single visual arrow to the mjvScene.
    This is a visual-only object and does not affect the physics.

    Args:
        scene (mjvScene): The scene to add the arrow to.
        from_point (np.ndarray): The starting point of the arrow.
        to_point (np.ndarray): The ending point of the arrow.
        radius (float): The radius of the arrow's shaft.
        rgba (tuple): The color and alpha of the arrow.
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

model = mujoco.MjModel.from_xml_path(mjcf_path)
model.opt.timestep = 1./1e3
model.dof_damping[-4:] = 3e-9
data = mujoco.MjData(model)
data.qpos[2] = 0.01  # Set initial position of the first joint
data.qacc[:] = 0  # Initialize accelerations to zero    
timestep = (model.opt.timestep 
            * 10
            )
model.opt.enableflags |= 1 << 0  # enable override
# solreflimit="4e-3 1" solimplimit=".95 .99 1e-3"
# model.opt.iterations = 200
model.opt.o_solref[0] = 4e-3
model.opt.o_solref[1] = 1
# self.model.opt.o_solref[1] = 5
model.opt.o_solimp[0] = 0.95 
model.opt.o_solimp[1] = 0.99 
model.opt.o_solimp[2] = 1e-3

pwm_freq = 1000
drive_freq = 20

points_per_period = pwm_freq // drive_freq
angles = np.linspace(0, 2 * np.pi, points_per_period, endpoint=False)
peak_trq = 2e-7

mujoco.mj_step(model, data)
data.qacc[:] = 0  # Initialize accelerations to zero    

rolls = []
pitches = []
yaws = []
jnts = []
angs = []

init_rs = []
for i in range(4):
    body_idx = i + 2
    body_quat = data.xquat[body_idx]
    body_frame = R.from_quat(body_quat)
    init_rs.append(deepcopy(body_frame))

# with mujoco.viewer.launch_passive(model, data) as viewer:
viewer = mujoco.viewer.launch_passive(model, data)
step_start = time.monotonic()
viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
viewer.cam.trackbodyid = 1


while viewer.is_running() and data.time < 50:
    print(data.time)
    viewer.user_scn.ngeom = 0

    angle = angles[int((data.time * pwm_freq) % points_per_period)]
    angs.append(angle)

    mag_norths = np.zeros((4, 3))
    goal_norths = np.zeros((4, 3))

    for i in range(4):
        body_idx = i + 2
        body_quat = data.xquat[body_idx]
        body_pos = data.xpos[body_idx]
        body_frame = R.from_quat(body_quat, scalar_first=True)
        arr_len = 0.01
        body_frame_dir = np.array([0,1,0])
        world_frame_dir = np.array([0,0,1])
        base_rot = init_rs[i]
        magnet_north = body_frame.as_matrix() @  body_frame_dir
        mag_norths[i] = magnet_north
        to = body_pos + arr_len * magnet_north

        rpy_rot = R.from_euler('y', angle, degrees=False)
        goal_north = rpy_rot.as_matrix() @ world_frame_dir
        goal_norths[i] = goal_north
        to_goal = body_pos + arr_len * goal_north

        add_visual_arrow(viewer.user_scn, body_pos[:3], to, rgba=(0, 1, 0, 1))
        add_visual_arrow(viewer.user_scn, body_pos[:3], to_goal, radius=0.0005, rgba=(1, 0, 0, 0.5))

        roll, pitch, yaw = body_frame.as_euler('zxy', degrees=False)
        kp_mag = 5e-6 * 1
        kv_mag = 1e-8 * 0

        data.xfrc_applied[body_idx,3:] = (kp_mag) * np.cross(magnet_north, goal_north) #-kv_mag * np.dot(magnet_north, goal_north)

    #     if i == 0:
    #         roll, pitch, yaw = body_frame.as_euler('zxy', degrees=False)
    #         rolls.append(roll)
    #         pitches.append(pitch)
    #         yaws.append(yaw)

    # jnts.append(data.qpos[7])     
    # print(f"acc: {data.qacc}")
    # print(f"vel: {data.qvel}")

    print(f"xfrc: {data.xfrc_applied}")

    mujoco.mj_step(model, data)
    viewer.sync()

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