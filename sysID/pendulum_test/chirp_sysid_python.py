"""
Magnetic Pendulum System Identification using Chirp Signal
==========================================================

This script performs system identification on a magnetic pendulum by:
1. Loading tracking data (angular position, velocity) and magnetic field measurements
2. Computing magnetic torque and friction forces based on physics model
3. Optimizing moment of inertia (J) and magnetic moment (B_magnet) parameters
4. Evaluating smoothness and single-valuedness of friction function

Physics Model:
- Magnetic torque: τ = B_magnet × B_ext × sin(θ_field - θ_track)
- Inertial torque: J × α (angular acceleration)
- Friction force: frc = τ - J × α (residual damping force)

Optimization finds J and B_magnet values that produce the smoothest,
most single-valued friction function of (θ, ω).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# Configuration
# ============================================================================

# File paths
TRACK_FILE = 'chirp-data/unlube3-1.csv'
FIELD_FILE = 'chirp3-unlube.csv'

# Field data trimming indices
B_START = 1528
B_END = 6521

# Resampling parameters
FS_NEW = 2000  # Hz
T_START = 0
T_END = 5
TRIM_END = 6521  # Number of points to keep after resampling

# Physical parameters (initial estimates)
MASS = 1.4e-5  # kg
DIAMETER = 2e-3  # m
J_INITIAL = 1/6 * MASS * DIAMETER**2  # Moment of inertia (kg⋅m²)
B_MAGNET_INITIAL = 0.00113  # Magnetic moment (A⋅m²)

# Optimization grid parameters
NG = 120  # Grid resolution
J_SEARCH_RANGE = (0.1, 1.5)  # Relative to J_INITIAL
B_SEARCH_RANGE = (0.1, 1.5)  # Relative to B_MAGNET_INITIAL
N_SEARCH_POINTS = 15  # Number of points in each dimension

# Smoothing parameters
GAUSSIAN_SIGMA = 1.0  # For surface smoothing

# Weighting parameter for combined objective
LAMBDA = 1.0  # Weight for multi-valuedness metric


# ============================================================================
# Step 1: Load and preprocess tracking data
# ============================================================================

def load_tracking_data(filepath):
    """Load tracking data with angular position and velocity."""
    track_tbl = pd.read_csv(filepath, header=1)

    t_trk = track_tbl['t'].values
    theta_trk = track_tbl['θ'].values / 180 * np.pi  # Convert to radians
    theta_trk = np.mod(theta_trk, 2*np.pi) - np.pi  # Wrap to [-π, π]
    omega_trk = track_tbl['ω'].values / 180 * np.pi  # Convert to rad/s

    # Check for and handle NaN values
    nan_count_omega = np.isnan(omega_trk).sum()
    nan_count_theta = np.isnan(theta_trk).sum()

    if nan_count_omega > 0 or nan_count_theta > 0:
        print(f"  NaN values found - Theta: {nan_count_theta}, Omega: {nan_count_omega}")

        if nan_count_omega > 0:
            omega_series = pd.Series(omega_trk)
            omega_series = omega_series.bfill().ffill()  # Backward fill first, then forward fill
            omega_trk = omega_series.values
            print(f"  Handled omega NaNs: {np.isnan(omega_trk).sum()} remaining")

        if nan_count_theta > 0:
            theta_series = pd.Series(theta_trk)
            theta_series = theta_series.bfill().ffill()
            theta_trk = theta_series.values
            print(f"  Handled theta NaNs: {np.isnan(theta_trk).sum()} remaining")

    return t_trk, theta_trk, omega_trk


def plot_tracking_data(t_trk, theta_trk, omega_trk):
    """Plot raw tracking data."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Theta over time
    axes[0, 0].plot(t_trk, theta_trk)
    axes[0, 0].axhline(np.mean(theta_trk), color='r', linestyle='--', label=f'Mean: {np.mean(theta_trk):.3f}')
    axes[0, 0].set_title('Theta over time')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Theta (rad)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Omega over time
    axes[0, 1].plot(t_trk, omega_trk)
    axes[0, 1].set_title('Omega over time')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Omega (rad/s)')
    axes[0, 1].grid(True)

    # Polar plot: theta vs omega
    ax_polar = plt.subplot(2, 2, 3, projection='polar')
    scatter = ax_polar.scatter(theta_trk, omega_trk, c=t_trk, s=5, cmap='viridis')
    ax_polar.set_ylim([np.min(omega_trk), np.max(omega_trk)])
    plt.colorbar(scatter, ax=ax_polar, label='Time (s)')
    ax_polar.set_title('Theta vs Omega (Polar)')

    plt.tight_layout()
    plt.show()


# ============================================================================
# Step 2: Load and preprocess magnetic field data
# ============================================================================

def load_field_data(filepath, start_idx, end_idx):
    """Load magnetic field measurements."""
    field_tbl = pd.read_csv(filepath, skiprows=6, sep=';')

    idx = field_tbl['Index'].values
    t_field = idx * 1e-3  # Convert ms to seconds
    t_field = t_field[start_idx:end_idx]

    Bx = field_tbl['Bx[mT]'].values[start_idx:end_idx]
    By = field_tbl['By[mT]'].values[start_idx:end_idx]
    Bz = field_tbl['Bz[mT]'].values[start_idx:end_idx]

    return t_field, Bx, By, Bz


def plot_field_data(t_field, Bx, By, Bz):
    """Plot magnetic field data."""
    fig = plt.figure(figsize=(14, 10))

    # 3D scatter of B field
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    scatter = ax1.scatter(Bx, By, Bz, c=t_field, s=10, cmap='viridis')
    ax1.set_xlabel('Bx (mT)')
    ax1.set_ylabel('By (mT)')
    ax1.set_zlabel('Bz (mT)')
    ax1.set_title('3D B field')
    plt.colorbar(scatter, ax=ax1, label='Time (s)')

    # By and Bz over time
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(By, label='By')
    ax2.plot(Bz, label='Bz')
    ax2.set_title('By and Bz')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Field (mT)')
    ax2.legend()
    ax2.grid(True)

    # Field angle over time
    theta_field = np.arctan2(Bz, By)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(theta_field)
    ax3.set_title('Angle over time')
    ax3.set_xlabel('Index')
    ax3.set_ylabel('Angle (rad)')
    ax3.grid(True)

    # Field magnitude over time
    mag_field = np.linalg.norm([By, Bz], axis=0)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(mag_field)
    ax4.set_title('Magnitude over time')
    ax4.set_xlabel('Index')
    ax4.set_ylabel('Magnitude (mT)')
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

    return theta_field, mag_field


# ============================================================================
# Step 3: Interpolate and synchronize data
# ============================================================================

def interpolate_and_sync_data(t_trk, theta_trk, omega_trk,
                              t_field, theta_field, mag_field,
                              fs_new, t_start, t_end, trim_end):
    """Interpolate all signals to common sampling rate and synchronize."""
    # Create new time vector
    t_new = np.arange(t_start, t_end + 1/fs_new, 1/fs_new)

    # Offset field time to start at zero
    t_field_offset = t_field - t_field[0]

    # Interpolate using cubic spline
    theta_field_spline = interp1d(t_field_offset, theta_field, kind='cubic',
                                  bounds_error=False, fill_value='extrapolate')(t_new)
    mag_field_spline = interp1d(t_field_offset, mag_field, kind='cubic',
                                bounds_error=False, fill_value='extrapolate')(t_new)
    omega_trk_spline = interp1d(t_trk, omega_trk, kind='cubic',
                                bounds_error=False, fill_value='extrapolate')(t_new)
    theta_trk_spline = interp1d(t_trk, theta_trk, kind='cubic',
                                bounds_error=False, fill_value='extrapolate')(t_new)

    # Check for NaN in interpolated data
    nan_checks = {
        'theta_field': np.isnan(theta_field_spline).sum(),
        'mag_field': np.isnan(mag_field_spline).sum(),
        'omega_trk': np.isnan(omega_trk_spline).sum(),
        'theta_trk': np.isnan(theta_trk_spline).sum()
    }
    if any(nan_checks.values()):
        print(f"  Warning: NaN values after interpolation:")
        for key, count in nan_checks.items():
            if count > 0:
                print(f"    {key}: {count} NaNs")

    # Compute angular acceleration via gradient
    alpha_trk_spline = np.gradient(omega_trk_spline, 1/fs_new)

    # Trim to specified length
    theta_field_trim = theta_field_spline[:trim_end]
    mag_field_trim = mag_field_spline[:trim_end]
    theta_trk_trim = theta_trk_spline[:trim_end]
    omega_trk_trim = omega_trk_spline[:trim_end]
    alpha_trk_trim = alpha_trk_spline[:trim_end]
    t_trim = t_new[:trim_end]

    return t_trim, theta_field_trim, mag_field_trim, theta_trk_trim, omega_trk_trim, alpha_trk_trim


def plot_synchronized_data(t_trim, theta_field_trim, theta_trk_trim):
    """Plot synchronized theta signals."""
    plt.figure(figsize=(10, 6))
    plt.plot(t_trim, theta_field_trim, label='Field', alpha=0.7)
    plt.plot(t_trim, theta_trk_trim, label='Track', alpha=0.7)
    plt.title('Synchronized Theta Signals')
    plt.xlabel('Time (s)')
    plt.ylabel('Theta (rad)')
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================================
# Step 4: Compute torque and friction force
# ============================================================================

def compute_forces(theta_field, theta_trk, mag_field, alpha_trk, J, B_magnet):
    """
    Compute magnetic torque and friction force.

    Parameters:
    -----------
    theta_field : array
        Magnetic field angle (rad)
    theta_trk : array
        Tracked pendulum angle (rad)
    mag_field : array
        Magnetic field magnitude (mT)
    alpha_trk : array
        Angular acceleration (rad/s²)
    J : float
        Moment of inertia (kg⋅m²)
    B_magnet : float
        Magnetic moment (A⋅m²)

    Returns:
    --------
    torque : array
        Magnetic torque (N⋅m)
    frc : array
        Friction force (N⋅m) - residual after subtracting inertial torque
    """
    B_ext = mag_field * 1e-3  # Convert mT to T
    B_const = B_magnet * B_ext
    torque = B_const * np.sin(theta_field - theta_trk)
    frc = torque - J * alpha_trk

    return torque, frc


def plot_forces(torque, frc):
    """Plot computed forces."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(torque)
    axes[0].set_title('Magnetic Torque')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Torque (N⋅m)')
    axes[0].grid(True)

    axes[1].plot(frc)
    axes[1].set_title('Friction Force')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Force (N⋅m)')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


# ============================================================================
# Step 5: Visualize friction in state space
# ============================================================================

def plot_friction_3d(theta_trk, omega_trk, frc, title="Friction in State Space"):
    """3D scatter plot of friction as function of (theta, omega)."""
    # Convert to Cartesian coordinates with radius offset
    omega_offset = omega_trk - np.min(omega_trk)
    xs = omega_offset * np.cos(theta_trk)
    ys = omega_offset * np.sin(theta_trk)
    zs = frc * 1e8  # Scale for visibility

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(xs, ys, zs, c=frc, s=10, cmap='viridis')
    ax.set_xlabel('ω cos(θ)')
    ax.set_ylabel('ω sin(θ)')
    ax.set_zlabel('Force × 10⁸ (N⋅m)')
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, label='Force (N⋅m)')

    # Equal aspect ratio
    max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min()]).max() / 2.0
    mid_x = (xs.max()+xs.min()) * 0.5
    mid_y = (ys.max()+ys.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)

    plt.show()


def plot_friction_polar(theta_trk, omega_trk, frc):
    """Polar plot of friction."""
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    scatter = ax.scatter(theta_trk, omega_trk, c=frc, s=5, cmap='viridis')
    ax.set_ylim([np.min(omega_trk), np.max(omega_trk)])
    plt.colorbar(scatter, ax=ax, label='Force (N⋅m)')
    ax.set_title('Friction: Theta vs Omega')
    plt.show()


# ============================================================================
# Step 6: Grid-based quality metrics
# ============================================================================

def compute_quality_metrics(theta_trk, omega_trk, frc, ng, gaussian_sigma):
    """
    Compute smoothness and multi-valuedness metrics for friction function.

    Parameters:
    -----------
    theta_trk : array
        Pendulum angles (rad)
    omega_trk : array
        Angular velocities (rad/s)
    frc : array
        Friction forces (N⋅m)
    ng : int
        Grid resolution
    gaussian_sigma : float
        Gaussian smoothing parameter

    Returns:
    --------
    smoothness : float
        Gradient energy (lower = smoother)
    multival : float
        Mean cell standard deviation (lower = more single-valued)
    nan_frac : float
        Fraction of NaN values in grid
    """
    from scipy.interpolate import griddata

    # Create grid
    theta_grid = np.linspace(0, 2*np.pi, ng)
    omega_grid = np.linspace(np.min(omega_trk), np.max(omega_trk), ng)
    TH, OM = np.meshgrid(theta_grid, omega_grid)

    # Wrap theta to [0, 2π] for gridding
    theta_wrapped = np.mod(theta_trk, 2*np.pi)

    # Grid the data using natural neighbor interpolation
    Fgrid = griddata(
        np.column_stack([theta_wrapped, omega_trk]),
        frc,
        (TH, OM),
        method='linear'
    )

    # Fill NaN holes with nearest neighbor
    nanmask = np.isnan(Fgrid)
    if np.any(nanmask):
        Fgrid_filled = griddata(
            np.column_stack([theta_wrapped, omega_trk]),
            frc,
            (TH[nanmask], OM[nanmask]),
            method='nearest'
        )
        Fgrid[nanmask] = Fgrid_filled

    # Smooth the grid
    Fgrid_s = gaussian_filter(Fgrid, sigma=gaussian_sigma)

    # --- Metric 1: Smoothness (gradient energy) ---
    dth = theta_grid[1] - theta_grid[0]
    dom = omega_grid[1] - omega_grid[0]
    dF_dtheta, dF_domega = np.gradient(Fgrid_s, dth, dom)

    valid_cells = np.sum(~np.isnan(Fgrid))
    if valid_cells > 0:
        Sg = np.sum(dF_dtheta**2 + dF_domega**2)
        Sg_norm = Sg / (valid_cells + 1e-10)
    else:
        Sg_norm = np.nan

    # --- Metric 2: Multi-valuedness (cell standard deviation) ---
    # Bin samples to grid cells
    ith = 1 + np.floor((theta_wrapped - theta_grid[0]) / dth).astype(int)
    iom = 1 + np.floor((omega_trk - omega_grid[0]) / dom).astype(int)

    # Clamp indices to valid range
    ith = np.clip(ith, 0, ng - 1)
    iom = np.clip(iom, 0, ng - 1)

    # Accumulate statistics per cell
    counts = np.zeros((ng, ng))
    sum1 = np.zeros((ng, ng))
    sum2 = np.zeros((ng, ng))

    for n in range(len(frc)):
        a, b = ith[n], iom[n]
        counts[b, a] += 1
        sum1[b, a] += frc[n]
        sum2[b, a] += frc[n]**2

    # Compute standard deviation per cell
    cell_std = np.full((ng, ng), np.nan)
    mask = counts > 1
    if np.any(mask):
        cell_std[mask] = np.sqrt(
            (sum2[mask] / counts[mask]) - (sum1[mask] / counts[mask])**2
        )
        Mult = np.nanmean(cell_std[mask])

        # Normalize by local scale
        local_scale = np.median(np.abs(frc))
        Mult_norm = Mult / (local_scale + 1e-10)
    else:
        Mult_norm = np.nan

    # Compute NaN fraction
    nan_frac = np.sum(np.isnan(Fgrid)) / Fgrid.size

    return Sg_norm, Mult_norm, nan_frac


# ============================================================================
# Step 7: Parameter optimization
# ============================================================================

def optimize_parameters(theta_field, theta_trk, mag_field, omega_trk, alpha_trk,
                       J_initial, B_magnet_initial,
                       J_range, B_range, n_points,
                       ng, gaussian_sigma, lambda_weight,
                       visualize=True):
    """
    Grid search over J and B_magnet to find optimal parameters.

    Returns:
    --------
    J_opt : float
        Optimal moment of inertia
    B_opt : float
        Optimal magnetic moment
    results : dict
        Contains metric arrays and search ranges
    """
    # Create search grids (log-spaced)
    J_search = np.logspace(
        np.log10(J_initial * J_range[0]),
        np.log10(J_initial * J_range[1]),
        n_points
    )
    B_search = np.logspace(
        np.log10(B_magnet_initial * B_range[0]),
        np.log10(B_magnet_initial * B_range[1]),
        n_points
    )

    # Initialize result arrays
    Smoothness = np.full((n_points, n_points), np.nan)
    Multival = np.full((n_points, n_points), np.nan)
    ObjMap = np.full((n_points, n_points), np.nan)

    print(f"Starting parameter optimization...")
    print(f"J search range: [{J_search[0]:.3e}, {J_search[-1]:.3e}] kg⋅m²")
    print(f"B search range: [{B_search[0]:.3e}, {B_search[-1]:.3e}] A⋅m²")

    for i, J_val in enumerate(J_search):
        for k, B_val in enumerate(B_search):
            # Compute forces for this parameter combination
            _, frc = compute_forces(theta_field, theta_trk, mag_field, alpha_trk, J_val, B_val)

            # Compute quality metrics
            Sg_norm, Mult_norm, nan_frac = compute_quality_metrics(
                theta_trk, omega_trk, frc, ng, gaussian_sigma
            )

            Smoothness[i, k] = Sg_norm
            Multival[i, k] = Mult_norm

            # Combined objective (lower is better)
            if np.isnan(Mult_norm):
                ObjMap[i, k] = Sg_norm
            else:
                ObjMap[i, k] = Sg_norm + lambda_weight * Mult_norm

            if visualize and (i * n_points + k) % 10 == 0:
                print(f"Progress: {i}/{n_points}, {k}/{n_points} | "
                      f"J={J_val/J_initial:.2f}×J₀, B={B_val/B_magnet_initial:.2f}×B₀ | "
                      f"Sg={Sg_norm:.3e}, Mult={Mult_norm:.3e}, NaN={nan_frac:.2%}")

    # Find optimal parameters (minimum combined objective)
    idx_opt = np.nanargmin(ObjMap)
    i_opt, k_opt = np.unravel_index(idx_opt, ObjMap.shape)
    J_opt = J_search[i_opt]
    B_opt = B_search[k_opt]

    print(f"\n{'='*60}")
    print(f"Optimal Parameters Found:")
    print(f"  J = {J_opt:.6e} kg⋅m² ({J_opt/J_initial:.3f} × J_initial)")
    print(f"  B = {B_opt:.6e} A⋅m² ({B_opt/B_magnet_initial:.3f} × B_initial)")
    print(f"  Smoothness = {Smoothness[i_opt, k_opt]:.6e}")
    print(f"  Multi-valuedness = {Multival[i_opt, k_opt]:.6e}")
    print(f"  Objective = {ObjMap[i_opt, k_opt]:.6e}")
    print(f"{'='*60}\n")

    results = {
        'J_search': J_search,
        'B_search': B_search,
        'Smoothness': Smoothness,
        'Multival': Multival,
        'ObjMap': ObjMap,
        'J_opt': J_opt,
        'B_opt': B_opt
    }

    return J_opt, B_opt, results


def plot_optimization_results(results, J_initial, B_initial):
    """Plot optimization metric heatmaps."""
    J_search = results['J_search']
    B_search = results['B_search']
    Smoothness = results['Smoothness']
    Multival = results['Multival']
    ObjMap = results['ObjMap']

    # Normalize to initial values for x/y axes
    J_norm = J_search / J_initial
    B_norm = B_search / B_initial

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Smoothness
    im1 = axes[0].imshow(Smoothness, aspect='auto', origin='lower',
                         extent=[B_norm[0], B_norm[-1], J_norm[0], J_norm[-1]],
                         cmap='viridis')
    axes[0].set_xlabel('B / B_initial')
    axes[0].set_ylabel('J / J_initial')
    axes[0].set_title('Smoothness (Gradient Energy)')
    plt.colorbar(im1, ax=axes[0])

    # Multi-valuedness
    im2 = axes[1].imshow(Multival, aspect='auto', origin='lower',
                         extent=[B_norm[0], B_norm[-1], J_norm[0], J_norm[-1]],
                         cmap='viridis')
    axes[1].set_xlabel('B / B_initial')
    axes[1].set_ylabel('J / J_initial')
    axes[1].set_title('Multi-valuedness (Cell Std)')
    plt.colorbar(im2, ax=axes[1])

    # Combined objective
    im3 = axes[2].imshow(ObjMap, aspect='auto', origin='lower',
                         extent=[B_norm[0], B_norm[-1], J_norm[0], J_norm[-1]],
                         cmap='viridis')
    axes[2].set_xlabel('B / B_initial')
    axes[2].set_ylabel('J / J_initial')
    axes[2].set_title('Combined Objective')
    plt.colorbar(im3, ax=axes[2])

    # Mark optimal point on all plots
    J_opt_norm = results['J_opt'] / J_initial
    B_opt_norm = results['B_opt'] / B_initial
    for ax in axes:
        ax.plot(B_opt_norm, J_opt_norm, 'r*', markersize=15, label='Optimal')
        ax.legend()

    plt.tight_layout()
    plt.show()


# ============================================================================
# Main execution
# ============================================================================

def main():
    """Main execution function."""
    print("="*60)
    print("Magnetic Pendulum System Identification")
    print("="*60)

    # Step 1: Load tracking data
    print("\n[1] Loading tracking data...")
    t_trk, theta_trk, omega_trk = load_tracking_data(TRACK_FILE)
    print(f"  Loaded {len(t_trk)} tracking samples")
    print(f"  Time range: [{t_trk[0]:.3f}, {t_trk[-1]:.3f}] s")
    plot_tracking_data(t_trk, theta_trk, omega_trk)

    # Step 2: Load field data
    print("\n[2] Loading magnetic field data...")
    t_field, Bx, By, Bz = load_field_data(FIELD_FILE, B_START, B_END)
    print(f"  Loaded {len(t_field)} field samples")
    print(f"  Time range: [{t_field[0]:.3f}, {t_field[-1]:.3f}] s")
    theta_field, mag_field = plot_field_data(t_field, Bx, By, Bz)

    # Step 3: Interpolate and synchronize
    print("\n[3] Interpolating and synchronizing data...")
    t_trim, theta_field_trim, mag_field_trim, theta_trk_trim, omega_trk_trim, alpha_trk_trim = \
        interpolate_and_sync_data(t_trk, theta_trk, omega_trk,
                                 t_field, theta_field, mag_field,
                                 FS_NEW, T_START, T_END, TRIM_END)
    print(f"  Resampled to {FS_NEW} Hz")
    print(f"  Final length: {len(t_trim)} samples")
    plot_synchronized_data(t_trim, theta_field_trim, theta_trk_trim)

    # Step 4: Compute initial forces
    print("\n[4] Computing forces with initial parameters...")
    print(f"  J_initial = {J_INITIAL:.6e} kg⋅m²")
    print(f"  B_magnet_initial = {B_MAGNET_INITIAL:.6e} A⋅m²")
    torque, frc = compute_forces(theta_field_trim, theta_trk_trim, mag_field_trim,
                                 alpha_trk_trim, J_INITIAL, B_MAGNET_INITIAL)
    plot_forces(torque, frc)

    # Step 5: Visualize friction
    print("\n[5] Visualizing friction in state space...")
    plot_friction_3d(theta_trk_trim, omega_trk_trim, frc,
                    title=f"Friction (J={J_INITIAL:.2e}, B={B_MAGNET_INITIAL:.2e})")
    plot_friction_polar(theta_trk_trim, omega_trk_trim, frc)

    # Step 6: Optimize parameters
    print("\n[6] Optimizing parameters...")
    J_opt, B_opt, results = optimize_parameters(
        theta_field_trim, theta_trk_trim, mag_field_trim, omega_trk_trim, alpha_trk_trim,
        J_INITIAL, B_MAGNET_INITIAL,
        J_SEARCH_RANGE, B_SEARCH_RANGE, N_SEARCH_POINTS,
        NG, GAUSSIAN_SIGMA, LAMBDA,
        visualize=True
    )

    # Step 7: Visualize optimization results
    print("\n[7] Visualizing optimization results...")
    plot_optimization_results(results, J_INITIAL, B_MAGNET_INITIAL)

    # Step 8: Compute forces with optimal parameters
    print("\n[8] Computing forces with optimal parameters...")
    torque_opt, frc_opt = compute_forces(theta_field_trim, theta_trk_trim, mag_field_trim,
                                        alpha_trk_trim, J_opt, B_opt)
    plot_friction_3d(theta_trk_trim, omega_trk_trim, frc_opt,
                    title=f"Friction (Optimal: J={J_opt:.2e}, B={B_opt:.2e})")

    print("\n" + "="*60)
    print("System Identification Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
