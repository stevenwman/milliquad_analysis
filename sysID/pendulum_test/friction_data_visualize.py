
import numpy as np
import pickle
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Physical parameters
J_INITIAL = 9.33e-12 
B_MAGNET_INITIAL = 0.00113 

def load_data(data_dir):
    try:
        with open(os.path.join(data_dir, 'data_unlubed_joint.pkl'), 'rb') as f:
            data_unlubed = pickle.load(f)
        with open(os.path.join(data_dir, 'data_lubed_joint.pkl'), 'rb') as f:
            data_lubed = pickle.load(f)
    except FileNotFoundError:
        data_dir = './'
        with open(os.path.join(data_dir, 'data_unlubed_joint.pkl'), 'rb') as f:
            data_unlubed = pickle.load(f)
        with open(os.path.join(data_dir, 'data_lubed_joint.pkl'), 'rb') as f:
            data_lubed = pickle.load(f)
            
    return data_unlubed, data_lubed

def calculate_friction(data):
    theta_trk = data['theta_trk_trim']
    alpha_trk = data['alpha_trk_trim']
    mag_field = data['mag_field_opt']
    theta_field = data['theta_field_opt']
    
    B_ext = mag_field * 1e-3 
    B_const = B_MAGNET_INITIAL * B_ext
    torque_mag = B_const * np.sin(theta_field - theta_trk)
    friction = torque_mag - J_INITIAL * alpha_trk
    
    return friction

def get_trace(data, name, omega_mask=None, cmin=None, cmax=None):
    theta = data['theta_trk_trim']
    omega = data['omega_trk_trim']
    friction = data['friction']
    
    # Filter by mask if provided
    if omega_mask is not None:
        theta = theta[omega_mask]
        omega = omega[omega_mask]
        friction = friction[omega_mask]
        
    if len(theta) == 0:
        return None

    # User Request: Center is 0, so radius is absolute velocity
    radius = np.abs(omega)
    
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = friction
    
    return go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=friction,
            colorscale='Viridis',
            cmin=cmin, cmax=cmax,
            opacity=0.8,
            showscale=True,
            colorbar=dict(title='Friction (Nm)', len=0.8)
        ),
        name=name
    )

def main():
    data_dir = '/home/sman/Work/CMU/Research/LEGO-milliquad-mujoco/sysID/pendulum_test/'
    print(f"Loading data from {data_dir}...")
    data_unlubed, data_lubed = load_data(data_dir)

    print("Calculating friction...")
    data_unlubed['friction'] = calculate_friction(data_unlubed)
    data_lubed['friction'] = calculate_friction(data_lubed)

    # Calculate global range for shared color scale
    all_fric = np.concatenate([data_unlubed['friction'], data_lubed['friction']])
    cmin, cmax = np.min(all_fric), np.max(all_fric)

    print("Generating 2x2 3D Grid Plot...")
    
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}],
               [{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("Unlubed (Omega > 0)", "Lubed (Omega > 0)", 
                        "Unlubed (Omega < 0)", "Lubed (Omega < 0)"),
        vertical_spacing=0.05
    )
    
    # Row 1: Positive Omega
    mask_u_pos = data_unlubed['omega_trk_trim'] > 0
    mask_l_pos = data_lubed['omega_trk_trim'] > 0
    
    fig.add_trace(get_trace(data_unlubed, "Unlubed (+)", mask_u_pos, cmin, cmax), row=1, col=1)
    fig.add_trace(get_trace(data_lubed, "Lubed (+)", mask_l_pos, cmin, cmax), row=1, col=2)
    
    # Row 2: Negative Omega
    mask_u_neg = data_unlubed['omega_trk_trim'] < 0
    mask_l_neg = data_lubed['omega_trk_trim'] < 0
    
    fig.add_trace(get_trace(data_unlubed, "Unlubed (-)", mask_u_neg, cmin, cmax), row=2, col=1)
    fig.add_trace(get_trace(data_lubed, "Lubed (-)", mask_l_neg, cmin, cmax), row=2, col=2)
    
    scene_layout = dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Friction'
    )
    
    fig.update_layout(
        title="Friction Phase Space: Unlubed vs Lubed | Positive vs Negative Omega",
        scene=scene_layout, scene2=scene_layout,
        scene3=scene_layout, scene4=scene_layout,
        height=1000,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    print("Opening 2x2 Plot...")
    fig.show()

if __name__ == "__main__":
    main()
