
import numpy as np
import pickle
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

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

def train_and_analyze(data, dataset_name, omega_mask=None, capture_output=False):
    # Buffer for output
    output_lines = []
    
    def log(msg):
        if capture_output:
            output_lines.append(msg)
        else:
            print(msg)

    # log("\n" + "="*80) # Skip header if side-by-side
    log(f"Models: {dataset_name}")
    log("="*60)
    
    theta = data['theta_trk_trim']
    omega = data['omega_trk_trim']
    y = data['friction']
    
    if omega_mask is not None:
        theta = theta[omega_mask]
        omega = omega[omega_mask]
        y = y[omega_mask]

    if len(y) == 0:
        return None, []

    # Centering
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    a = np.mean(sin_theta)
    b = np.mean(cos_theta)
    c = np.mean(omega)
    
    S_c = sin_theta - a
    C_c = cos_theta - b
    W_c = omega - c
    
    X_base = np.column_stack((S_c, C_c, W_c))

    models_out = {} 
    best_full_r2 = -np.inf
    best_add_r2 = -np.inf

    log(f"{'Type':<25} | {'Ord':<3} | {'Feat':<4} | {'R2':<6}")
    log("-" * 45)

    for order in range(1, 5): 
        # Full Poly
        poly = PolynomialFeatures(degree=order, include_bias=False)
        X_full = poly.fit_transform(X_base)
        feature_names_full = poly.get_feature_names_out(['S_c', 'C_c', 'W_c'])
        
        model_full = LinearRegression()
        model_full.fit(X_full, y)
        y_pred_full = model_full.predict(X_full)
        r2_full = r2_score(y, y_pred_full)
        
        log(f"{'Full Poly':<25} | {order:<3} | {X_full.shape[1]:<4} | {r2_full:.4f}")

        if r2_full > best_full_r2:
            best_full_r2 = r2_full
            models_out['best_full'] = {
                'model': model_full, 'X': X_full, 'names': feature_names_full, 'pred': y_pred_full, 'order': order
            }

        # Additive Poly
        if order == 1:
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
            
        log(f"{'Additive':<25} | {order:<3} | {X_additive.shape[1]:<4} | {r2_add:.4f}")

        if r2_add > best_add_r2:
            best_add_r2 = r2_add
            models_out['best_add'] = {
                'model': model_add, 'X': X_additive, 'names': feature_names_add, 'pred': y_pred_add, 'order': order
            }
        
        if order == 2:
            models_out['ord2_full'] = {
                'model': model_full, 'X': X_full, 'names': feature_names_full, 'pred': y_pred_full, 'order': order
            }
            models_out['ord2_add'] = {
                'model': model_add, 'X': X_additive, 'names': feature_names_add, 'pred': y_pred_add, 'order': order
            }
            
        if order == 1:
            models_out['linear'] = {
                'model': model_full, 'X': X_full, 'names': feature_names_full, 'pred': y_pred_full, 'order': order
            }

    # Helper Analysis Function
    def analyze(model_struct, name, min_contrib=1e-9):
        model = model_struct['model']
        names = model_struct['names']
        X = model_struct['X']
        
        log(f"\n{name} (R2={model.score(X, y):.4f})")
        log("-" * 60)
        
        y_pred = model.predict(X)
        mse_orig = np.mean((y - y_pred)**2)
        stats = []
        
        if model.fit_intercept:
            contrib = np.full_like(y, model.intercept_)
            mse_no_bias = np.mean((y - (y_pred - model.intercept_))**2)
            pct_error_inc = ((mse_no_bias - mse_orig) / mse_orig) * 100
            stats.append({"Term": "Intercept", "Coeff": model.intercept_, "AvgAbsContrib": np.mean(np.abs(contrib)), "PctErrorInc": pct_error_inc})
            
        for i, term_name in enumerate(names):
            coef = model.coef_[i]
            contrib = coef * X[:, i]
            y_pred_no_term = y_pred - contrib
            mse_no_term = np.mean((y - y_pred_no_term)**2)
            pct_error_inc = ((mse_no_term - mse_orig) / mse_orig) * 100
            stats.append({"Term": term_name, "Coeff": coef, "AvgAbsContrib": np.mean(np.abs(contrib)), "PctErrorInc": pct_error_inc})
            
        stats.sort(key=lambda x: x["AvgAbsContrib"], reverse=True)
        
        # Condensed header for side-by-side
        log(f"{'Term':<15} | {'Coeff':<10} | {'Contrib':<10} | {'%Eff':<5}")
        log("-" * 50)
        count = 0 
        for s in stats:
            if s["AvgAbsContrib"] > min_contrib:
                # shorten numbers
                log(f"{s['Term']:<15} | {s['Coeff']:<10.2e} | {s['AvgAbsContrib']:<10.2e} | {s['PctErrorInc']:<5.1f}")
                count += 1
                if count >= 10: # Limit lines per table for side-by-side readability?
                    pass 
        log("-" * 50)

    # Generate analysis text blocks
    analyze(models_out['best_full'], f"Best Full", min_contrib=1e-9) # Use cutoff to save space
    analyze(models_out['ord2_full'], f"Ord 2 Full", min_contrib=0.0)
    analyze(models_out['linear'], f"Linear", min_contrib=0.0)
    
    return models_out, output_lines

def print_side_by_side(lines_left, lines_right, width=60):
    max_len = max(len(lines_left), len(lines_right))
    print(f"\n{'Left Column':<{width}} | {'Right Column'}")
    print("-" * (width * 2 + 3))
    
    for i in range(max_len):
        l = lines_left[i] if i < len(lines_left) else ""
        r = lines_right[i] if i < len(lines_right) else ""
        # Truncate if too long to avoid wrapping
        l = (l[:width-3] + "..") if len(l) > width else l
        print(f"{l:<{width}} | {r}")

def get_trace(data, name, omega_mask, y_data=None, color='data', cmin=None, cmax=None):
    theta = data['theta_trk_trim']
    omega = data['omega_trk_trim']
    
    theta = theta[omega_mask]
    omega = omega[omega_mask]
    
    if y_data is None:
        z = data['friction'][omega_mask]
    else:
        z = y_data
        
    if len(theta) == 0:
        return None

    radius = np.abs(omega)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    marker_dict = dict(size=2, opacity=0.8)
    
    if color == 'data':
        marker_dict.update(dict(
            color=z,
            colorscale='Viridis',
            cmin=cmin, cmax=cmax,
            showscale=True,
            colorbar=dict(title='Friction (Nm)', len=0.8)
        ))
        marker_dict['opacity'] = 0.3 
    else:
        marker_dict.update(dict(color=color, opacity=0.5))

    return go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=marker_dict,
        name=name,
        showlegend=True
    )

def main():
    data_dir = '/home/sman/Work/CMU/Research/LEGO-milliquad-mujoco/sysID/pendulum_test/'
    print(f"Loading data from {data_dir}...")
    data_unlubed, data_lubed = load_data(data_dir)

    print("Calculating friction...")
    data_unlubed['friction'] = calculate_friction(data_unlubed)
    data_lubed['friction'] = calculate_friction(data_lubed)

    all_fric = np.concatenate([data_unlubed['friction'], data_lubed['friction']])
    cmin, cmax = np.min(all_fric), np.max(all_fric)

    mask_u_pos = data_unlubed['omega_trk_trim'] > 0
    mask_u_neg = data_unlubed['omega_trk_trim'] < 0
    mask_l_pos = data_lubed['omega_trk_trim'] > 0
    mask_l_neg = data_lubed['omega_trk_trim'] < 0
    
    print("\n" + "="*120)
    print("ANALYSIS: POSITIVE vs NEGATIVE OMEGA")
    print("="*120)

    # 1. Compare Unlubed (+) vs Unlubed (-)
    print(f"\n>>> UNLUBED: Positive (Left) vs Negative (Right)")
    models_u_pos, logs_u_pos = train_and_analyze(data_unlubed, "Unlubed (+)", mask_u_pos, capture_output=True)
    models_u_neg, logs_u_neg = train_and_analyze(data_unlubed, "Unlubed (-)", mask_u_neg, capture_output=True)
    print_side_by_side(logs_u_pos, logs_u_neg, width=65)

    # 2. Compare Lubed (+) vs Lubed (-)
    print(f"\n>>> LUBED: Positive (Left) vs Negative (Right)")
    models_l_pos, logs_l_pos = train_and_analyze(data_lubed, "Lubed (+)", mask_l_pos, capture_output=True)
    models_l_neg, logs_l_neg = train_and_analyze(data_lubed, "Lubed (-)", mask_l_neg, capture_output=True)
    print_side_by_side(logs_l_pos, logs_l_neg, width=65)

    
    # Helper to generate grid plot
    def create_grid_plot(model_key, plot_title):
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'scene'}],
                   [{'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=("Unlubed (+)", "Lubed (+)", "Unlubed (-)", "Lubed (-)"),
            vertical_spacing=0.05
        )
        
        # Row 1: Positive
        fig.add_trace(get_trace(data_unlubed, "Data U(+)", mask_u_pos, color='data', cmin=cmin, cmax=cmax), row=1, col=1)
        if model_key: fig.add_trace(get_trace(data_unlubed, "Model U(+)", mask_u_pos, y_data=models_u_pos[model_key]['pred'], color='red'), row=1, col=1)
        
        fig.add_trace(get_trace(data_lubed, "Data L(+)", mask_l_pos, color='data', cmin=cmin, cmax=cmax), row=1, col=2)
        if model_key: fig.add_trace(get_trace(data_lubed, "Model L(+)", mask_l_pos, y_data=models_l_pos[model_key]['pred'], color='orange'), row=1, col=2)

        # Row 2: Negative
        fig.add_trace(get_trace(data_unlubed, "Data U(-)", mask_u_neg, color='data', cmin=cmin, cmax=cmax), row=2, col=1)
        if model_key: fig.add_trace(get_trace(data_unlubed, "Model U(-)", mask_u_neg, y_data=models_u_neg[model_key]['pred'], color='red'), row=2, col=1)
        
        fig.add_trace(get_trace(data_lubed, "Data L(-)", mask_l_neg, color='data', cmin=cmin, cmax=cmax), row=2, col=2)
        if model_key: fig.add_trace(get_trace(data_lubed, "Model L(-)", mask_l_neg, y_data=models_l_neg[model_key]['pred'], color='orange'), row=2, col=2)
        
        scene_layout = dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Friction')
        fig.update_layout(title=plot_title, scene=scene_layout, scene2=scene_layout, scene3=scene_layout, scene4=scene_layout, height=1000)
        return fig

    print("\nGenerating 2x2 Grid Plots...")
    create_grid_plot(None, "Original Data (Segregated)").show()
    create_grid_plot('linear', "Linear Model Comparison (Segregated)").show()
    create_grid_plot('ord2_full', "Order 2 Full Comparison (Segregated)").show()
    create_grid_plot('best_full', "Best Full Model Comparison (Segregated)").show()
    
    print("Done.")

if __name__ == "__main__":
    main()
