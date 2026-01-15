# Magnetic Pendulum System Identification

Python translation of MATLAB chirp system identification code for magnetic pendulum experiments.

## Overview

This code performs system identification on a magnetic pendulum by optimizing physical parameters (moment of inertia J and magnetic moment B) to produce the smoothest, most single-valued friction function.

## Physics Model

The system is modeled using:
- **Magnetic torque**: τ = B_magnet × B_ext × sin(θ_field - θ_track)
- **Inertial torque**: J × α (angular acceleration)
- **Friction force**: frc = τ - J × α (residual damping force)

## Files

1. **chirp_sysid_python.ipynb** (Recommended)
   - Interactive Jupyter notebook with step-by-step execution
   - Includes visualizations and detailed explanations
   - Best for exploration and understanding

2. **chirp_sysid_python.py**
   - Standalone Python script
   - Can be run from command line
   - Same functionality as notebook

## Data Synchronization: Splicing and Interpolation

The tracking data and magnetic field data come from different sensors with different sampling rates and time bases. To combine them for analysis, we perform a two-step synchronization process:

### Step 1: Splicing (Temporal Alignment)

**Problem**: The two data sources have different time ranges and may not start at the same moment.

**Solution**:
1. **Field data splicing**: Extract relevant time window using `B_START` and `B_END` indices
   - Original field data: ~7000 samples
   - After splicing: samples [1528:6521] → 4993 samples
   - This selects the time window where the chirp signal is active

2. **Time offset alignment**:
   ```python
   t_field_offset = t_field - t_field[0]  # Shift field time to start at 0
   ```
   - Makes both datasets share a common time reference

### Step 2: Interpolation (Rate Harmonization)

**Problem**: The two data sources have different, non-uniform sampling rates:
- Tracking data: Variable rate (~200 Hz typical)
- Field data: 1 kHz (1 sample per millisecond)

**Solution**: Resample everything to a uniform 2 kHz grid using cubic spline interpolation.

```python
# Create uniform time grid
t_new = np.arange(0, 5, 1/2000)  # 0 to 5 seconds at 2kHz

# Interpolate all signals to this grid
theta_field_interp = interp1d(t_field_offset, theta_field, kind='cubic')(t_new)
theta_trk_interp = interp1d(t_trk, theta_trk, kind='cubic')(t_new)
omega_trk_interp = interp1d(t_trk, omega_trk, kind='cubic')(t_new)
mag_field_interp = interp1d(t_field_offset, mag_field, kind='cubic')(t_new)
```

**Why cubic spline?**
- Smooth interpolation preserves derivatives (important for computing angular acceleration)
- Continuous first and second derivatives
- No artificial oscillations between data points

### Step 3: Final Trimming

After interpolation, trim to desired length:
```python
# Trim to 6521 samples (3.26 seconds at 2kHz)
all_signals = all_signals[:6521]
```

This ensures all signals have exactly the same:
- ✅ Number of samples
- ✅ Sample rate (2 kHz)
- ✅ Time base (0 to 3.26 seconds)

### Visual Summary

```
Before Synchronization:
┌─────────────────────────────────────────────────────────┐
│ Tracking Data:                                          │
│   • Variable rate (~200 Hz)                            │
│   • 10070 samples over 5.03 seconds                    │
│   • Time: [0.0000, 5.0345] s                           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Field Data (full):                                      │
│   • Fixed rate (1 kHz)                                  │
│   • 7088 samples                                        │
│   • Time: [0.001, 7.088] s                             │
└─────────────────────────────────────────────────────────┘

After Splicing:
┌─────────────────────────────────────────────────────────┐
│ Field Data (spliced):                                   │
│   • Samples [1528:6521] extracted                       │
│   • 4993 samples                                        │
│   • Time offset to [0, 4.993] s                        │
└─────────────────────────────────────────────────────────┘

After Interpolation:
┌─────────────────────────────────────────────────────────┐
│ All Signals (synchronized):                             │
│   • Uniform rate: 2000 Hz                              │
│   • 6521 samples                                        │
│   • Time: [0.0000, 3.2605] s                           │
│   • All signals time-aligned sample-by-sample          │
└─────────────────────────────────────────────────────────┘
```

### Computing Angular Acceleration

After synchronization, we can compute angular acceleration using the gradient:

```python
alpha_trk = np.gradient(omega_trk_interp, 1/2000)  # rad/s²
```

This works because:
1. Uniform sampling → constant time step (Δt = 0.0005 s)
2. Smooth interpolation → no artificial noise in derivative
3. High sample rate (2 kHz) → accurate gradient estimation

## Data Files

### Input Data Required:
- **Tracking data**: `chirp-data/unlube3-1.csv`
  - Contains: time (t), position (x, y), angle (θ), angular velocity (ω)
  - Format: CSV with 2 header rows

- **Field data**: `chirp3-unlube.csv`
  - Contains: magnetic field measurements (Bx, By, Bz)
  - Format: CSV with 6 header rows, semicolon-separated

### Data Quality:
- NaN values in tracking data (first/last points) are automatically handled via forward/backward fill
- Data synchronization achieved through splicing and interpolation (see below)

## Usage

### Jupyter Notebook (Recommended):
```bash
cd /home/sman/Work/CMU/Research/LEGO-milliquad-mujoco/sysID/pendulum_test
jupyter notebook chirp_sysid_python.ipynb
```

Then run cells sequentially to:
1. Load and visualize tracking data
2. Load and visualize magnetic field data
3. Synchronize and interpolate data to 2kHz
4. Compute forces with initial parameters
5. Visualize friction in state space
6. Run parameter optimization
7. Compare initial vs optimal parameters

### Python Script:
```bash
cd /home/sman/Work/CMU/Research/LEGO-milliquad-mujoco/sysID/pendulum_test
python3 chirp_sysid_python.py
```

## Configuration Parameters

Key parameters that can be adjusted in the notebook/script:

```python
# Physical parameters (initial estimates)
MASS = 1.4e-5  # kg
DIAMETER = 2e-3  # m
J_INITIAL = 1/6 * MASS * DIAMETER**2  # Moment of inertia
B_MAGNET_INITIAL = 0.00113  # Magnetic moment (A⋅m²)

# Optimization parameters
NG = 120  # Grid resolution
J_SEARCH_RANGE = (0.1, 1.5)  # Search range relative to J_INITIAL
B_SEARCH_RANGE = (0.1, 1.5)  # Search range relative to B_INITIAL
N_SEARCH_POINTS = 15  # Grid points per dimension

# Smoothing
GAUSSIAN_SIGMA = 1.0  # Surface smoothing parameter
LAMBDA = 1.0  # Weight for multi-valuedness metric
```

## Optimization Metrics

The algorithm evaluates two quality metrics:

1. **Smoothness** (Sg_norm): Gradient energy of gridded friction surface
   - Measures how smooth the friction function is
   - Lower is better

2. **Multi-valuedness** (Mult_norm): Mean standard deviation within grid cells
   - Measures how single-valued the friction function is
   - Lower is better

**Combined Objective**: `Obj = Smoothness + λ × Multi-valuedness`

The optimal parameters minimize this combined objective.

## Output

The code produces:
- Polar plots of (θ, ω) phase space
- 3D scatter plots of friction surface
- Heatmaps of optimization metrics
- Side-by-side comparison of initial vs optimal parameters
- Optimal parameter values with quality metrics

## NaN Handling

The code automatically handles:
- NaN values in tracking data (typically at first/last measurement points)
- Uses backward-fill then forward-fill strategy
- Checks for NaNs after interpolation
- Reports any data quality issues

## Troubleshooting

**Issue**: "FileNotFoundError" for CSV files
- Check that you're in the correct directory
- Verify file paths in configuration section

**Issue**: Too many NaN values after interpolation
- Check that time ranges in tracking and field data overlap
- Adjust T_START, T_END, or TRIM_END parameters

**Issue**: Optimization takes too long
- Reduce N_SEARCH_POINTS (try 10 instead of 15)
- Reduce NG grid resolution (try 80 instead of 120)

## Differences from MATLAB Code

Key improvements in Python version:
1. Proper CSV parsing with correct headers/delimiters
2. Automatic NaN handling with reporting
3. Progress indicators during optimization
4. More detailed documentation
5. Interactive notebook format for exploration
