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
    # The magnetic torque tries to align the magnet with the field
    torque_mag = B_const * np.sin(theta_field - theta_trk)
    
    # Friction = Torque_applied - J * alpha
    # Equation of motion: J*alpha = Torque_mag - Torque_friction
    # So Torque_friction = Torque_mag - J*alpha
    friction = torque_mag - J_INITIAL * alpha_trk
    
    return friction, torque_mag

def plot_friction_cylindrical(data, title_suffix=""):
    theta = data['theta_trk_trim']
    omega = data['omega_trk_trim']
    friction = data['friction']
    
    # ---------------------------------------------------------
    # 3D Cylindrical Plotting
    # ---------------------------------------------------------
    # We will visualize the Friction as a function of Angle and Velocity.
    # Cylindrical Coordinates Interpretation:
    # Angle (theta) -> Angle around the z-axis
    # Velocity (omega) -> Radius from z-axis? 
    # Friction -> Z-axis height?
    # 
    # Alternatively, to strictly strictly follow "Results in 3D cylindrical coordinates":
    # One might map:
    # x = Friction * cos(theta)
    # y = Friction * sin(theta)
    # z = Omega
    # But usually "Cylindrical Plot" means domain is polar.
    # Let's assume Domain = (Theta, Omega), Range = Friction.
    # So we map (Theta, Omega) -> (x, y). 
    # x = Omega * cos(Theta)
    # y = Omega * sin(Theta)
    # z = Friction
    #
    # PROBLEM: Omega can be negative. Radius must be positive.
    # Solution: Shift Omega by min(omega) or use abs(omega). 
    # Usually phase space is (theta, omega). 
    # Using r = omega - min(omega) ensures positivity.
    
    omega_min = np.min(omega)
    radius = omega - omega_min  # This ensures radius >= 0
    # Add a small buffer if needed to avoid radius exactly 0 at one point, but >=0 is sufficient for visualization distance
    
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = friction
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=friction,                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8,
            colorbar=dict(title='Friction (Nm)')
        )
    )])
    
    fig.update_layout(
        title=f"Friction Torque in Phase Space {title_suffix} (Cylindrical: r=omega-min(omega), theta=angle, z=friction)",
        scene=dict(
            xaxis_title='(Omega - Omega_min) * cos(Theta)',
            yaxis_title='(Omega - Omega_min) * sin(Theta)',
            zaxis_title='Friction Torque (Nm)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

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

    # Add results to dataframe or dict for plotting convenience
    data_unlubed['friction'] = friction_unlubed
    data_lubed['friction'] = friction_lubed

    print(f"Unlubed Friction Range: [{np.min(friction_unlubed):.2e}, {np.max(friction_unlubed):.2e}] Nm")
    print(f"Lubed Friction Range:   [{np.min(friction_lubed):.2e}, {np.max(friction_lubed):.2e}] Nm")

    print("Generating 3D plots...")
    
    fig1 = plot_friction_cylindrical(data_unlubed, "(Unlubed)")
    print("Opening Unlubed Plot...")
    fig1.show()

    fig2 = plot_friction_cylindrical(data_lubed, "(Lubed)")
    print("Opening Lubed Plot...")
    fig2.show()


    # -------------------------------------------------------------
    # Step 2: Polynomial Modeling (Unlubed)
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("Fitting Polynomial Models (Unlubed Data)")
    print("="*50)
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import PolynomialFeatures

    # Prepare features and target
    # Feature vector X_raw: [sin(theta), cos(theta), omega]
    theta = data_unlubed['theta_trk_trim']
    omega = data_unlubed['omega_trk_trim']
    y = data_unlubed['friction']

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # "first order term can be (sin-a)^n * (cos-b)^n * (omega-c)^n" implies centering
    a = np.mean(sin_theta)
    b = np.mean(cos_theta)
    c = np.mean(omega)
    
    # Centered features
    S_c = sin_theta - a
    C_c = cos_theta - b
    W_c = omega - c
    
    X_base = np.column_stack((S_c, C_c, W_c)) # Shape (N, 3)

    results = []
    
    # Track best models
    best_full_model = None
    best_full_r2 = -np.inf
    best_full_features = None
    best_full_feature_names = None
    best_full_pred = None
    
    best_add_model = None
    best_add_r2 = -np.inf
    best_add_features = None
    best_add_feature_names = None
    best_add_pred = None

    # Track Order 2 models specifically (User Request)
    ord2_full_model = None
    ord2_full_features = None
    ord2_full_names = None
    ord2_full_pred = None
    
    ord2_add_model = None
    ord2_add_features = None
    ord2_add_names = None
    ord2_add_pred = None

    print(f"{'Type':<25} | {'Order':<5} | {'Features':<8} | {'R2 Score':<10}")
    print("-" * 60)

    for order in range(1, 5): # 1, 2, 3, 4
        # --- Strategy 1: Full Polynomial ---
        poly = PolynomialFeatures(degree=order, include_bias=False)
        X_full = poly.fit_transform(X_base)
        feature_names_full = poly.get_feature_names_out(['S_c', 'C_c', 'W_c'])
        
        model_full = LinearRegression()
        model_full.fit(X_full, y)
        y_pred_full = model_full.predict(X_full)
        r2_full = r2_score(y, y_pred_full)
        
        results.append({
            "Type": "Full Poly (Cross Terms)",
            "Order": order,
            "R2": r2_full,
            "Num_Features": X_full.shape[1]
        })
        print(f"{'Full Poly (Cross Terms)':<25} | {order:<5} | {X_full.shape[1]:<8} | {r2_full:.6f}")

        if r2_full > best_full_r2:
            best_full_r2 = r2_full
            best_full_model = model_full
            best_full_features = X_full
            best_full_feature_names = feature_names_full
            best_full_pred = y_pred_full

        # --- Strategy 2: Additive Polynomial (No Cross Terms) ---
        if order == 1:
            # Order 1 Additive is same as Full
            r2_add = r2_full
            X_additive = X_full
            feature_names_add = feature_names_full
            model_add = model_full
            y_pred_add = y_pred_full
        else:
            X_add_list = []
            feature_names_add = []
            for power in range(1, order + 1):
                X_add_list.append(X_base**power) 
                feature_names_add.extend([f"S_c^{power}", f"C_c^{power}", f"W_c^{power}"])
                
            X_additive = np.hstack(X_add_list)
            
            model_add = LinearRegression()
            model_add.fit(X_additive, y)
            y_pred_add = model_add.predict(X_additive)
            r2_add = r2_score(y, y_pred_add)
            
        results.append({
            "Type": "Additive (No Cross Terms)",
            "Order": order,
            "R2": r2_add,
            "Num_Features": X_additive.shape[1]
        })
        print(f"{'Additive (No Cross Terms)':<25} | {order:<5} | {X_additive.shape[1]:<8} | {r2_add:.6f}")

        if r2_add > best_add_r2:
            best_add_r2 = r2_add
            best_add_model = model_add
            # best_add_features = X_additive
            best_add_feature_names = feature_names_add
            best_add_pred = y_pred_add
        
        # Capture Order 2
        if order == 2:
            ord2_full_model = model_full
            ord2_full_features = X_full
            ord2_full_names = feature_names_full
            ord2_full_pred = y_pred_full
            
            ord2_add_model = model_add
            ord2_add_features = X_additive
            ord2_add_names = feature_names_add
            ord2_add_pred = y_pred_add

    # Helper to print detailed analysis
    def analyze_model_terms(model, feature_names, X, y, model_name, min_contrib=1e-9):
        print(f"\n{'='*80}")
        print(f"Detailed Analysis: {model_name} (R2={model.score(X, y):.4f})")
        print(f"{'='*80}")
        
        y_pred_orig = model.predict(X)
        mse_orig = np.mean((y - y_pred_orig)**2)
        
        # Collect stats
        stats = []
        
        # Intercept (Bias)
        if model.fit_intercept:
             # Contribution is constant
            contrib = np.full_like(y, model.intercept_)
            mean_abs_contrib = np.mean(np.abs(contrib))
            
            # Error if removed
            y_pred_no_bias = y_pred_orig - model.intercept_
            mse_no_bias = np.mean((y - y_pred_no_bias)**2)
            pct_error_inc = ((mse_no_bias - mse_orig) / mse_orig) * 100
            
            stats.append({
                "Term": "Intercept",
                "Coeff": model.intercept_,
                "AvgAbsContrib": mean_abs_contrib,
                "PctErrorInc": pct_error_inc
            })
            
        # Features
        for i, name in enumerate(feature_names):
            coef = model.coef_[i]
            term_values = X[:, i]
            contrib = coef * term_values
            
            # Metric 1: Average Absolute Contribution
            mean_abs_contrib = np.mean(np.abs(contrib))
            
            # Metric 2: Error Increase if term removed (zeroed out)
            # y_new = y_old - (coef * x_i)
            # This is faster than re-predicting everything
            y_pred_no_term = y_pred_orig - contrib
            mse_no_term = np.mean((y - y_pred_no_term)**2)
            pct_error_inc = ((mse_no_term - mse_orig) / mse_orig) * 100
            
            stats.append({
                "Term": name,
                "Coeff": coef,
                "AvgAbsContrib": mean_abs_contrib,
                "PctErrorInc": pct_error_inc
            })
            
        # Sort by AvgAbsContrib descending
        stats.sort(key=lambda x: x["AvgAbsContrib"], reverse=True)
        
        # Print Table
        print(f"{'Term':<30} | {'Coeff':<12} | {'Avg Abs Contrib (Nm)':<22} | {'% MSE Incr (if rm)':<20}")
        print("-" * 90)
        for s in stats:
            # Only print meaningful terms (e.g. contribution > min_contrib)
            if s["AvgAbsContrib"] > min_contrib:
                print(f"{s['Term']:<30} | {s['Coeff']:<12.4e} | {s['AvgAbsContrib']:<22.4e} | {s['PctErrorInc']:<20.2f}")
        print("-" * 90)

    # Analyze Best Models
    analyze_model_terms(best_full_model, best_full_feature_names, best_full_features, y, "Best Full Poly Model")
    
    # Analyze Best Additive Model
    # Re-construct X_additive specific to the best order
    best_add_order_found = -1
    for i, res in enumerate(results):
        if "Additive" in res['Type'] and abs(res['R2'] - best_add_r2) < 1e-9:
            best_add_order_found = res['Order']
            break
            
    if best_add_order_found > 0:
        X_add_list = []
        for power in range(1, best_add_order_found + 1):
            X_add_list.append(X_base**power) 
        X_additive_best = np.hstack(X_add_list)
        analyze_model_terms(best_add_model, best_add_feature_names, X_additive_best, y, "Best Additive Model")

    # Analyze Order 2 Models
    # User Request: "all term printout for the second order fits", so min_contrib=0.0
    if ord2_full_model:
        analyze_model_terms(ord2_full_model, ord2_full_names, ord2_full_features, y, "Order 2 Full Poly Model", min_contrib=0.0)
    if ord2_add_model:
        analyze_model_terms(ord2_add_model, ord2_add_names, ord2_add_features, y, "Order 2 Additive Model", min_contrib=0.0)


    # -------------------------------------------------------------
    # Step 3: Plotting Best Models vs Data
    # -------------------------------------------------------------
    print("\nGenerating Model Comparison Plots...")
    
    # Helper to plot comparison
    def plot_model_comparison(data, y_pred_model, model_name):
        theta = data['theta_trk_trim']
        omega = data['omega_trk_trim']
        friction_data = data['friction']
        
        omega_min = np.min(omega)
        radius = omega - omega_min
        
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        fig = go.Figure()
        
        # Trace 1: Original Data
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=friction_data,
            mode='markers',
            marker=dict(size=2, color='blue', opacity=0.3),
            name='Measured Data'
        ))
        
        # Trace 2: Model Prediction
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=y_pred_model,
            mode='markers',
            marker=dict(size=2, color='red', opacity=0.5),
            name=f'Model ({model_name})'
        ))
        
        fig.update_layout(
            title=f"Unlubed Friction: Data vs {model_name}",
            scene=dict(
                xaxis_title='(Omega-min)*cos(theta)',
                yaxis_title='(Omega-min)*sin(theta)',
                zaxis_title='Friction Torque'
            )
        )
        return fig

    # Plot Best Full Model
    print("Opening Best Full Model Plot...")
    fig_full = plot_model_comparison(data_unlubed, best_full_pred, "Best Full Poly (Ord 4)")
    fig_full.show()

    # Plot Best Additive Model
    print("Opening Best Additive Model Plot...")
    fig_add = plot_model_comparison(data_unlubed, best_add_pred, "Best Additive (Ord 4)")
    fig_add.show()
    
    # Plot Order 2 Full Model
    print("Opening Order 2 Full Model Plot...")
    fig_ord2_full = plot_model_comparison(data_unlubed, ord2_full_pred, "Order 2 Full Poly")
    fig_ord2_full.show()

    # Plot Order 2 Additive Model
    print("Opening Order 2 Additive Model Plot...")
    fig_ord2_add = plot_model_comparison(data_unlubed, ord2_add_pred, "Order 2 Additive")
    fig_ord2_add.show()

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()