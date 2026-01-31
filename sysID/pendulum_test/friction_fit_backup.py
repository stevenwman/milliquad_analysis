
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
        print(f"File not found in {data_dir}. Trying local.")
        with open('data_unlubed_joint.pkl', 'rb') as f:
            data_unlubed = pickle.load(f)
        with open('data_lubed_joint.pkl', 'rb') as f:
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

def train_and_analyze(data, dataset_name):
    print("\n" + "="*80)
    print(f"Fitting Polynomial Models ({dataset_name})")
    print("="*80)
    
    theta = data['theta_trk_trim']
    omega = data['omega_trk_trim']
    y = data['friction']

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

    results_summary = []
    
    # Storage for key models
    models_out = {} # keys: 'best_full', 'best_add', 'ord2_full', 'ord2_add'

    best_full_r2 = -np.inf
    best_add_r2 = -np.inf

    print(f"{'Type':<25} | {'Order':<5} | {'Features':<8} | {'R2 Score':<10}")
    print("-" * 60)

    for order in range(1, 5): 
        # Full Poly
        poly = PolynomialFeatures(degree=order, include_bias=False)
        X_full = poly.fit_transform(X_base)
        feature_names_full = poly.get_feature_names_out(['S_c', 'C_c', 'W_c'])
        
        model_full = LinearRegression()
        model_full.fit(X_full, y)
        y_pred_full = model_full.predict(X_full)
        r2_full = r2_score(y, y_pred_full)
        
        print(f"{'Full Poly':<25} | {order:<5} | {X_full.shape[1]:<8} | {r2_full:.6f}")

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
            # Construct additive manually
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
            
        print(f"{'Additive':<25} | {order:<5} | {X_additive.shape[1]:<8} | {r2_add:.6f}")

        if r2_add > best_add_r2:
            best_add_r2 = r2_add
            models_out['best_add'] = {
                'model': model_add, 'X': X_additive, 'names': feature_names_add, 'pred': y_pred_add, 'order': order
            }
        
        # Capture Order 2
        if order == 2:
            models_out['ord2_full'] = {
                'model': model_full, 'X': X_full, 'names': feature_names_full, 'pred': y_pred_full, 'order': order
            }
            models_out['ord2_add'] = {
                'model': model_add, 'X': X_additive, 'names': feature_names_add, 'pred': y_pred_add, 'order': order
            }

        # Capture Linear (Order 1) - using Full is fine as per user
        if order == 1:
            models_out['linear'] = {
                'model': model_full, 'X': X_full, 'names': feature_names_full, 'pred': y_pred_full, 'order': order
            }

    # Analysis Helper
    def analyze(model_struct, name, min_contrib=1e-9):
        model = model_struct['model']
        names = model_struct['names']
        X = model_struct['X']
        
        print(f"\n{'='*80}")
        print(f"Detailed Analysis: {name} (R2={model.score(X, y):.4f})")
        print(f"{'='*80}")
        
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
        print(f"{'Term':<30} | {'Coeff':<12} | {'Avg Abs Contrib (Nm)':<22} | {'% MSE Incr (if rm)':<20}")
        print("-" * 90)
        for s in stats:
            if s["AvgAbsContrib"] > min_contrib:
                print(f"{s['Term']:<30} | {s['Coeff']:<12.4e} | {s['AvgAbsContrib']:<22.4e} | {s['PctErrorInc']:<20.2f}")
        print("-" * 90)

    # Perform Analysis
    analyze(models_out['best_full'], f"Best Full Model ({dataset_name})")
    analyze(models_out['best_add'], f"Best Additive Model ({dataset_name})")
    analyze(models_out['ord2_full'], f"Order 2 Full Model ({dataset_name})", min_contrib=0.0)
    analyze(models_out['ord2_add'], f"Order 2 Additive Model ({dataset_name})", min_contrib=0.0)
    analyze(models_out['linear'], f"Linear Model ({dataset_name})", min_contrib=0.0)
    
    return models_out


def create_side_by_side_plot(data_u, pred_u, data_l, pred_l, title, cmin, cmax):
    # Setup subplots: 1 Row, 2 Cols
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("Unlubed", "Lubed")
    )

    # Data Coords Unlubed
    tu, ou, fu = data_u['theta_trk_trim'], data_u['omega_trk_trim'], data_u['friction']
    ru = ou - np.min(ou)
    xu, yu = ru * np.cos(tu), ru * np.sin(tu)

    # Data Coords Lubed
    tl, ol, fl = data_l['theta_trk_trim'], data_l['omega_trk_trim'], data_l['friction']
    rl = ol - np.min(ol)
    xl, yl = rl * np.cos(tl), rl * np.sin(tl)

    # --- UNLUBED PLOT (Left) ---
    # Data
    fig.add_trace(go.Scatter3d(
        x=xu, y=yu, z=fu, mode='markers',
        marker=dict(
            size=2, 
            color=fu,               # Map color to friction value
            colorscale='Viridis',   
            cmin=cmin, cmax=cmax,   # Global Scale
            opacity=0.3,
            showscale=True,         # Show colorbar
            colorbar=dict(title='Friction (Nm)', x=-0.1, len=0.8) # Position to left
        ),
        name='Data (Unlubed)', showlegend=True 
    ), row=1, col=1)
    
    # Model (if provided)
    if pred_u is not None:
        fig.add_trace(go.Scatter3d(
            x=xu, y=yu, z=pred_u, mode='markers',
            marker=dict(size=2, color='red', opacity=0.5),
            name='Prediction', showlegend=True
        ), row=1, col=1)

    # --- LUBED PLOT (Right) ---
    # Data
    fig.add_trace(go.Scatter3d(
        x=xl, y=yl, z=fl, mode='markers',
        marker=dict(
            size=2, 
            color=fl,               # Map color to friction value
            colorscale='Viridis',
            cmin=cmin, cmax=cmax,   # Global Scale   
            opacity=0.3,
            showscale=True,         # Show colorbar
            colorbar=dict(title='Friction (Nm)', x=1.1, len=0.8)
        ),
        name='Data (Lubed)', showlegend=True
    ), row=1, col=2)

    # Model (if provided)
    if pred_l is not None:
        fig.add_trace(go.Scatter3d(
            x=xl, y=yl, z=pred_l, mode='markers',
            marker=dict(size=2, color='orange', opacity=0.5),
            name='Prediction', showlegend=True
        ), row=1, col=2)

    # Layout
    scene_common = dict(
        xaxis_title='(Omega-min)*cos(Theta)',
        yaxis_title='(Omega-min)*sin(Theta)',
        zaxis_title='Friction (Nm)'
    )
    fig.update_layout(title=title, scene=scene_common, scene2=scene_common)
    return fig

def main():
    data_dir = '/home/sman/Work/CMU/Research/LEGO-milliquad-mujoco/sysID/pendulum_test/'
    data_unlubed, data_lubed = load_data(data_dir)
    
    data_unlubed['friction'] = calculate_friction(data_unlubed)
    data_lubed['friction'] = calculate_friction(data_lubed)

    print(f"Unlubed Range: [{np.min(data_unlubed['friction']):.2e}, {np.max(data_unlubed['friction']):.2e}]")
    print(f"Lubed Range:   [{np.min(data_lubed['friction']):.2e}, {np.max(data_lubed['friction']):.2e}]")
    
    # Calculate Global Color Scale Limits for consistency
    cmin = min(np.min(data_unlubed['friction']), np.min(data_lubed['friction']))
    cmax = max(np.max(data_unlubed['friction']), np.max(data_lubed['friction']))

    # Fit Models
    models_unlubed = train_and_analyze(data_unlubed, "Unlubed")
    models_lubed = train_and_analyze(data_lubed, "Lubed")

    print("\nGenerating Side-by-Side Plots...")
    
    # 1. Original Data Comparison
    fig1 = create_side_by_side_plot(
        data_unlubed, None, 
        data_lubed, None,
        "Original Data: Unlubed vs Lubed",
        cmin, cmax
    )
    fig1.show()

    # 2. Best Full Model
    fig2 = create_side_by_side_plot(
        data_unlubed, models_unlubed['best_full']['pred'],
        data_lubed, models_lubed['best_full']['pred'],
        "Best Full Poly Model Comparison",
        cmin, cmax
    )
    fig2.show()

    # 3. Best Additive Model
    fig3 = create_side_by_side_plot(
        data_unlubed, models_unlubed['best_add']['pred'],
        data_lubed, models_lubed['best_add']['pred'],
        "Best Additive Model Comparison",
        cmin, cmax
    )
    fig3.show()

    # 4. Order 2 Full
    fig4 = create_side_by_side_plot(
        data_unlubed, models_unlubed['ord2_full']['pred'],
        data_lubed, models_lubed['ord2_full']['pred'],
        "Order 2 Full Model Comparison",
        cmin, cmax
    )
    fig4.show()

    # 5. Order 2 Additive
    fig5 = create_side_by_side_plot(
        data_unlubed, models_unlubed['ord2_add']['pred'],
        data_lubed, models_lubed['ord2_add']['pred'],
        "Order 2 Additive Model Comparison",
        cmin, cmax
    )
    fig5.show()

    # 6. Linear Model (Order 1)
    fig6 = create_side_by_side_plot(
        data_unlubed, models_unlubed['linear']['pred'],
        data_lubed, models_lubed['linear']['pred'],
        "Linear Model Comparison",
        cmin, cmax
    )
    fig6.show()
    
    print("Done.")

if __name__ == "__main__":
    main()