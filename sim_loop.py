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

    # Get a reference to the next available geom in the scene
    geom = scene.geoms[scene.ngeom]
    
    # Set the properties of the arrow
    mujoco.mjv_initGeom(geom, type=mujoco.mjtGeom.mjGEOM_ARROW,
                        size=np.array([radius, radius, np.linalg.norm(to_point - from_point)]),
                        pos=np.zeros(3), mat=np.eye(3).flatten(), # Will be updated by mjv_connector
                        rgba=np.array(rgba, dtype=np.float32))
    
    # Use MuJoCo's built-in function to correctly position and orient the arrow.
    # This version uses mjv_connector and passes the full numpy arrays.
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 
                         radius, from_point, to_point)

    # Increment the number of geoms in the scene
    scene.ngeom += 1
    # print(scene.ngeom, "geoms in scene")


model = mujoco.MjModel.from_xml_path(mjcf_path)
data = mujoco.MjData(model)
data.qpos[0] = 0.01  # Set initial position of the first joint

timestep = model.opt.timestep * 30

model.opt.enableflags |= 1 << 0  # enable override
# solreflimit="4e-3 1" solimplimit=".95 .99 1e-3"
# self.model.opt.iterations = 200
model.opt.o_solref[0] = 4e-3
model.opt.o_solref[1] = 1
# self.model.opt.o_solref[1] = 5

model.opt.o_solimp[0] = 0.95 
model.opt.o_solimp[1] = 0.99 
model.opt.o_solimp[2] = 1e-3

pwm_freq= 1000

drive_freq = 40

points_per_period = pwm_freq // drive_freq
angles = np.linspace(0, 2 * np.pi, points_per_period, endpoint=False)
peak_trq = 2e-7

mujoco.mj_step(model, data)

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

with mujoco.viewer.launch_passive(model, data) as viewer:
  # Record the start time of the first step.
    step_start = time.monotonic()

    while viewer.is_running() and data.time < 50:
        viewer.user_scn.ngeom = 0

        angle = angles[int((data.time * pwm_freq) % points_per_period)]
        angs.append(angle)

        for i in range(4):
            body_idx = i + 2
            body_quat = data.xquat[body_idx]
            body_pos = data.xpos[body_idx]
            body_frame = R.from_quat(body_quat, scalar_first=True)
            arr_len = 0.01
            body_frame_dir = np.array([0,1,0])
            world_frame_dir = np.array([0,0,1])
            base_rot = init_rs[i]
            to = body_pos + arr_len * body_frame.as_matrix() @  body_frame_dir

            rpy_rot = R.from_euler('y', angle, degrees=False)

            to_goal = body_pos + arr_len * rpy_rot.as_matrix() @ world_frame_dir

            add_visual_arrow(viewer.user_scn, body_pos[:3], to, rgba=(0, 1, 0, 1))
            add_visual_arrow(viewer.user_scn, body_pos[:3], to_goal, radius=0.0005, rgba=(1, 0, 0, 0.5))

            roll, pitch, yaw = body_frame.as_euler('zxy', degrees=False)
            ang_error = np.array([angle - roll, roll - angle])
            ang_error_abs = np.min(np.abs(ang_error))

            error_sign = 1 if ang_error[0] < ang_error[1] else -1

            kp_mag = 1.5e-7

            data.xfrc_applied[i,:] = - kp_mag * ang_error_abs * error_sign * np.array([0, 0, 0, 0, 1, 0])

            if i == 0:
                roll, pitch, yaw = body_frame.as_euler('zxy', degrees=False)
                rolls.append(roll)
                pitches.append(pitch)
                yaws.append(yaw)

        jnts.append(data.qpos[7])
        
        # wheg_wrench = peak_trq * np.array([0, 0, 0, 0, 1, 0])

        # data.xfrc_applied[2,:] = wheg_wrench
        # data.xfrc_applied[3,:] = wheg_wrench
        # data.xfrc_applied[4,:] = wheg_wrench
        # data.xfrc_applied[5,:] = wheg_wrench
      

        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = timestep - (time.monotonic() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        step_start = time.monotonic()

plt.figure()
tot_plots = 5
plt.subplot(tot_plots, 1, 1)
plt.plot(np.arange(len(rolls)) * timestep, rolls)
plt.title('Roll')
plt.subplot(tot_plots, 1, 2)
plt.plot(np.arange(len(pitches)) * timestep, pitches)
plt.title('Pitch')
plt.subplot(tot_plots, 1, 3)
plt.plot(np.arange(len(yaws)) * timestep, yaws)
plt.title('Yaw')
plt.subplot(tot_plots, 1, 4)
plt.plot(np.arange(len(angs)) * timestep, angs)
plt.title('Target Roll Angle')
plt.subplot(tot_plots, 1, 5)
plt.plot(np.arange(len(jnts)) * timestep, np.array(jnts) % (2 * np.pi))
plt.title('Joint Position')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.savefig('orientation_plot.png') # Save the plot to a file
print("Plot saved to orientation_plot.png")