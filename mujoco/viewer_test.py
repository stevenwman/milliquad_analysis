import mujoco
import mujoco.viewer
import time
import numpy as np

# A minimal MJCF model with a spinning box. No external files are needed.
xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1.5"/>
    <body name="box_body" pos="0 0 0.3">
      <joint name="box_joint" type="hinge" axis="0 0 1"/>
      <geom name="box" type="box" size=".1 .1 .1" rgba="0.8 0.2 0.2 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="spin_motor" joint="box_joint" gear="0.25"/>
  </actuator>
</mujoco>
"""

# Load the model and create the data instance.
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)


def render_text_overlay(viewport, context):
    """
    This function is called every frame to draw the text overlay.
    It receives the viewport and rendering context as arguments.
    """
    # --- Display Simulation Time ---
    time_text = f"Time: {data.time:.2f} s"
    mujoco.mjr_text(
        font=mujoco.mjtFont.mjFONT_NORMAL,
        txt=time_text,
        context=context,
        x=10,  # X-position in pixels from the left
        y=viewport.height - 30,  # Y-position in pixels from the bottom
        r=1.0, g=1.0, b=1.0  # White color
    )

    # --- Display Actuator Control Value ---
    ctrl_text = f"Control Input: {data.ctrl[0]:.2f}"
    mujoco.mjr_text(
        font=mujoco.mjtFont.mjFONT_NORMAL,
        txt=ctrl_text,
        context=context,
        x=10,  # X-position in pixels from the left
        y=viewport.height - 55,  # Y-position below the first line
        r=0.5, g=0.8, b=1.0  # Light blue color
    )

# --- Main Simulation ---

# Launch the passive viewer.
# The 'with' statement ensures the viewer is properly closed.
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Assign the callback function to the viewer's overlay.
    # This is done ONCE, before the simulation loop starts.
    viewer.user_scn.overlay = render_text_overlay

    # The main simulation loop.
    while viewer.is_running():
        step_start = time.time()

        # Apply a sinusoidal control signal to the motor.
        data.ctrl[0] = np.sin(data.time * 2)

        # Advance the simulation by one step.
        mujoco.mj_step(model, data)

        # Update the viewer to reflect the new simulation state.
        viewer.sync()

        # Rudimentary real-time synchronization to make the simulation
        # run at a speed close to real-time.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)