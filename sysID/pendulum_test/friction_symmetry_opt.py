
import numpy as np
import pickle
import os
import plotly.graph_objects as go
from scipy.optimize import minimize
import plotly.express as px

# Physical parameters (Initial Guesses)
J_INITIAL = 9.33e-12 
B_MAGNET_INITIAL = 0.00113 

def load_data(data_dir):
    try:
        with open(os.path.join(data_dir, 'data_unlubed_joint.pkl'), 'rb') as f:
            data = pickle.load(f) # Only optimizing on Unlubed for now (cleaner physics)
    except FileNotFoundError:
        data_dir = './'
        with open(os.path.join(data_dir, 'data_unlubed_joint.pkl'), 'rb') as f:
            data = pickle.load(f)
    return data

def get_friction(params, data):
    k_J, k_B = params
    
    theta_trk = data['theta_trk_trim']
    alpha_trk = data['alpha_trk_trim']
    mag_field = data['mag_field_opt']
    theta_field = data['theta_field_opt']
    
    B_ext = mag_field * 1e-3 
    # Apply Scaling Factor k_B
    B_const = (B_MAGNET_INITIAL * k_B) * B_ext
    
    torque_mag = B_const * np.sin(theta_field - theta_trk)
    
    # Apply Scaling Factor k_J
    # Friction = Torque_mag - J * alpha
    friction = torque_mag - (J_INITIAL * k_J) * alpha_trk
    
    return friction

def symmetry_loss(params, data):
    """
    Loss function: Measures how well the derived friction fits a perfectly symmetric behavior.
    Assumption: Friction depends largely on velocity and maybe position, but F(v) should be roughly -F(-v).
    
    Implementation:
    Fit a simple model: F_pred = sign(w) * (C1 + C2*|w|) 
    This model forces symmetry. The residual variance is the "hysteresis" or asymmetry.
    """
    
    friction = get_friction(params, data)
    omega = data['omega_trk_trim']
    
    # Simple Symmetric Model: Viscous + Coulomb
    # F = w * C_viscous + sign(w) * C_coulomb
    # Solve via Least Squares: A x = b
    
    # Features matrix
    # Col 0: Omega (Viscous)
    # Col 1: Sign(Omega) (Coulomb)
    # Col 2: Bias (Should be near zero if symmetric, but included to absorb offset)
    X = np.column_stack((omega, np.sign(omega))) #, np.ones_like(omega)))
    
    # Regularized Least squares
    # (X.T X)^-1 X.T y
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(X, friction, rcond=None)
        
        # Calculate MSE of the residual
        f_sym = X @ coeffs
        mse = np.mean((friction - f_sym)**2)
        
        # PREVENT TRIVIAL SOLUTION (Restored because user wants to find non-zero sign errors)
        total_variance = np.var(friction)
        
        if total_variance < 1e-15:
            return 1.0 # High penalty for vanishing signal
            
        sym_score = mse / total_variance 
        
        return sym_score
        
    except Exception as e:
        return 1e9

def main():
    data_dir = '/home/sman/Work/CMU/Research/LEGO-milliquad-mujoco/sysID/pendulum_test/'
    print(f"Loading Unlubed data...")
    data = load_data(data_dir)

    print("\nStarting Optimization (Minimizing Hysteresis)...")
    print("Initial Params: kJ=1.00, kB=1.00")
    
    # Bounds: Check negative space for sign errors. [-10, 10] reasonable range.
    bounds = [(-10.0, 10.0), (-10.0, 10.0)] 
    initial_guess = [1.0, 1.0]

    # Use Nelder-Mead for robustness (deriv free)
    result = minimize(
        symmetry_loss, 
        initial_guess, 
        args=(data,), 
        method='Nelder-Mead', 
        bounds=bounds,
        tol=1e-6
    )

    k_J_opt, k_B_opt = result.x
    final_loss = result.fun
    
    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Optimal Scale Factors:")
    print(f"  k_J (Inertia): {k_J_opt:.4f}")
    print(f"  k_B (Magnet):  {k_B_opt:.4f}")
    
    # Bound Check
    if np.isclose(k_J_opt, -10.0) or np.isclose(k_J_opt, 10.0):
        print("WARNING: k_J hit the optimization boundary!")
    if np.isclose(k_B_opt, -10.0) or np.isclose(k_B_opt, 10.0):
        print("WARNING: k_B hit the optimization boundary!")
        
    print(f"Final Symmetry Loss: {final_loss:.2e}")
    
    print(f"\nNew Physical Constants:")
    print(f"  J_opt: {J_INITIAL * k_J_opt:.4e} kg.m^2")
    print(f"  B_opt: {B_MAGNET_INITIAL * k_B_opt:.4e} A.m^2")
    
    # --- VISUALIZATION ---
    print("\nGenerating Comparison Plots...")
    
    f_old = get_friction([1.0, 1.0], data)
    f_new = get_friction([k_J_opt, k_B_opt], data)
    omega = data['omega_trk_trim']
    
    fig = go.Figure()
    
    # 1. Old Parameters
    fig.add_trace(go.Scatter(
        x=omega, y=f_old,
        mode='markers',
        marker=dict(size=2, color='red', opacity=0.3),
        name='Original Params'
    ))
    
    # 2. New Parameters
    fig.add_trace(go.Scatter(
        x=omega, y=f_new,
        mode='markers',
        marker=dict(size=2, color='blue', opacity=0.3),
        name=f'Optimized (kJ={k_J_opt:.2f}, kB={k_B_opt:.2f})'
    ))
    
    fig.update_layout(
        title="Friction vs Velocity: Optimization Effect (Symmetry Check)",
        xaxis_title="Omega (rad/s)",
        yaxis_title="Friction Torque (Nm)",
        template='plotly_white'
    )
    
    fig.show()

    # --- LOSS LANDSCAPE (Linear Scale across Zero) ---
    print("generating loss landscape (linear +/- space)...")
    
    # Grid: span from -10 to 10
    resolution = 50
    x_range = np.linspace(-10.0, 10.0, resolution)
    y_range = np.linspace(-10.0, 10.0, resolution)
    
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = symmetry_loss([X[i,j], Y[i,j]], data)
            
    fig2 = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig2.update_layout(
        title="Loss Landscape (Probing Sign Errors)",
        scene = dict(
            xaxis_title='k_J',
            yaxis_title='k_B',
            zaxis_title='Loss (Normalized MSE)'
        )
    )
    fig2.show()
    
if __name__ == "__main__":
    main()
