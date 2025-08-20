import os
import mujoco
import mujoco.viewer


file_path = "one_milli_quad/scene.xml"

# Load your model
model = mujoco.MjModel.from_xml_path(file_path)
# Create a simulation data structure
data = mujoco.MjData(model)
data.qpos[0] = 0.01  # Set initial position of the first joint
# Launch the viewer (GUI)
mujoco.viewer.launch(model, data)