
import numpy as np
import pickle
import os

import plotly.graph_objects as go

# Physical parameters extracted from chirp_data_preprocessing_sync_backup.ipynb
J_INITIAL = 9.33e-12  # Moment of inertia (kg⋅m²)
B_MAGNET_INITIAL = 0.00113  # Magnetic moment (A⋅m²)

def load_data(data_dir):
    file_unlubed = os.path.join(data_dir, 'data_unlubed_joint.pkl')
    file_lubed = os.path.join(data_dir, 'data_lubed_joint.pkl')

    with open(file_unlubed, 'rb') as f:
        data_unlubed = pickle.load(f)

    with open(file_lubed, 'rb') as f:
        data_lubed = pickle.load(f)
        
    return data_unlubed, data_lubed

def calculate_friction(data):
    # Extract scalar fields
    # Keys: 'theta_trk_trim', 'omega_trk_trim', 'mag_field_opt', 'theta_field_opt', 'alpha_trk_trim'
    theta_trk = data['theta_trk_trim']
    alpha_trk = data['alpha_trk_trim']
    mag_field = data['mag_field_opt']
    theta_field = data['theta_field_opt']
    
    # Calculate Magnetic Gradient Torque
    # B_ext needs to be in Tesla (file has mT likely, based on 1e-3 factor in source)
    B_ext = mag_field * 1e-3 
    B_const = B_MAGNET_INITIAL * B_ext
    
    # Torque = mu * B * sin(delta_theta)
    torque_mag = B_const * np.sin(theta_field - theta_trk)
    
    # Friction = Torque_applied - J * alpha
    friction = torque_mag - J_INITIAL * alpha_trk
    
    return friction, torque_mag

def plot_friction_cylindrical(data, title_suffix=""):
    theta = data['theta_trk_trim']
    omega = data['omega_trk_trim']
    friction = data['friction']
    

    
    # Mapping to Phase Cylinder projection in Cartesian coords for 3D plot
    # User Request: Radius must be positive => Shift omega by min(omega)
    # Radius = Omega_shifted
    # Angle = Theta
    # Z = Friction
    
    omega_min = np.min(omega)
    radius = omega - omega_min  # This ensures radius >= 0
    # Add a small buffer if needed to avoid radius exactly 0 at one point, but >=0 is sufficient for visualization distance
    
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = friction
    
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=friction,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='Friction (Nm)')
        ),
        name=f'Friction {title_suffix}'
    )

def main():
    data_dir = '/home/sman/Work/CMU/Research/LEGO-milliquad-mujoco/sysID/pendulum_test/'
    print(f"Loading data from {data_dir}...")
    try:
        data_unlubed, data_lubed = load_data(data_dir)
    except FileNotFoundError:
        # Fallback to current directory if absolute path invalid in this context
        data_dir = './'
        data_unlubed, data_lubed = load_data(data_dir)

    print("Calculating friction...")
    friction_unlubed, torque_unlubed = calculate_friction(data_unlubed)
    friction_lubed, torque_lubed = calculate_friction(data_lubed)

    # Store results
    data_unlubed['friction'] = friction_unlubed
    data_lubed['friction'] = friction_lubed

    print(f"Unlubed Friction Range: [{np.min(friction_unlubed):.2e}, {np.max(friction_unlubed):.2e}] Nm")
    print(f"Lubed Friction Range:   [{np.min(friction_lubed):.2e}, {np.max(friction_lubed):.2e}] Nm")

    print("Generating 3D plots...")
    
    # Plot 1: Unlubed
    fig1 = go.Figure()
    fig1.add_trace(plot_friction_cylindrical(data_unlubed, "Unlubed"))
    fig1.update_layout(
        title="Unlubed Friction Torque in Phase Space (Cylindrical: r=omega-min(omega), theta=angle, z=friction)",
        scene=dict(
            xaxis_title='(Omega - Omega_min) * cos(Theta)',
            yaxis_title='(Omega - Omega_min) * sin(Theta)',
            zaxis_title='Friction Torque (Nm)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    print("Opening Unlubed Plot...")
    fig1.show()

    # Plot 2: Lubed
    fig2 = go.Figure()
    fig2.add_trace(plot_friction_cylindrical(data_lubed, "Lubed"))
    fig2.update_layout(
        title="Lubed Friction Torque in Phase Space (Cylindrical: r=omega-min(omega), theta=angle, z=friction)",
        scene=dict(
            xaxis_title='(Omega - Omega_min) * cos(Theta)',
            yaxis_title='(Omega - Omega_min) * sin(Theta)',
            zaxis_title='Friction Torque (Nm)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    print("Opening Lubed Plot...")
    fig2.show()


if __name__ == "__main__":
    main()
