import numpy as np
import pickle
import os
from scipy.interpolate import griddata

# Load data
data_dir = '/home/sman/Work/CMU/Research/LEGO-milliquad-mujoco/sysID/pendulum_test/'
with open(os.path.join(data_dir, 'data_unlubed_joint.pkl'), 'rb') as f:
    data = pickle.load(f)

theta = data['theta_trk_trim']
omega = data['omega_trk_trim']
# Mock friction (random) just to test grid coverage
friction = np.random.randn(*theta.shape)

# Create grid
resolution = 100
t_min, t_max = np.min(theta), np.max(theta)
o_min, o_max = np.min(omega), np.max(omega)

grid_t, grid_o = np.meshgrid(
    np.linspace(t_min, t_max, resolution),
    np.linspace(o_min, o_max, resolution)
)

points = np.column_stack((theta, omega))
grid_z = griddata(points, friction, (grid_t, grid_o), method='linear')

total_points = grid_z.size
nan_points = np.isnan(grid_z).sum()
print(f"Total Grid Points: {total_points}")
print(f"NaN Points: {nan_points} ({nan_points/total_points*100:.2f}%)")
