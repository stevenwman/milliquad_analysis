# Why Scene2 (2-spoke) and Scene4 (4-spoke) Do Well on Rough Terrain

## Background

The "scene #" refers to the number of spokes on a leg connected to the hip joint:
- **Scene1**: Single leg (standard single-spoke configuration)
- **Scene2**: 2 legs 180° apart (2-spoke)
- **Scene4**: 4 spokes 90° apart (4-spoke)
- **Wheel**: Continuous cylindrical contact (note: MuJoCo model uses 4 discrete mesh bodies with the same joint topology as spoke robots — body IDs 2-5 map to 4 wheel segments, so per-leg contact data represents per-segment data for wheel)

The robot is ~6mm, magnetically actuated via an external rotating field. Rough terrain has ~1mm std roughness — a 17% obstacle-to-body ratio, right in the transition zone where locomotion strategy matters most.

---

## Hypotheses

### H1. Contact Redundancy / Duty Factor

More spokes = more contact opportunities per revolution. On rough terrain where any single footfall might land on an unfavorable asperity, having 2 or 4 spokes at different angular phases provides backup. Scene1 has ~180° dead zones between contacts; scene4 shrinks that to ~45°.

**Metric: Duty factor** — fraction of timesteps where at least one leg geom contacts terrain, per morphology. Higher duty factor = more continuous propulsion.

**Metric: Per-leg contact count** — time series of how many legs are simultaneously in contact. Scene4 should show higher simultaneous contact counts.

**Interpretive framework**:
- If scene4 duty factor >> scene1 duty factor on rough terrain → supports contact redundancy hypothesis
- Compare flat vs rough: if duty factor drops more for scene1 than scene4 on rough → multi-spoke is robust to terrain
- Simultaneous contact count histogram: scene4 should peak at 2-3 simultaneous contacts, scene1 at 0-1
- **Wheel as limit case**: Wheel's cylindrical mesh geometry should produce higher duty factor than spoke designs, though the 4-body discretization in MuJoCo means it won't be a perfect continuous contact — expect high but not necessarily 100%. If scene4 duty factor is already near wheel levels, contact redundancy saturates early. If wheel duty factor drops on rough terrain despite smooth geometry, discrete spokes may offer terrain-adaptive advantages that continuous surfaces don't.

Literature: Whegs research (Quinn et al. 2003), HAMR insect-scale robot (Goldberg et al. 2018).

### H2. Slip vs Grip Balance

At this scale, controlled slipping may help. Multi-spoke designs distribute propulsive force across more contacts, keeping each below the slip threshold more often.

**Metric: Slip fraction** — for each leg-terrain contact, compute `F_tangent / F_normal`. If this ratio equals the friction coefficient `mu`, the contact is slipping. Report fraction of contact-timesteps that are at the friction cone boundary.

**What we expect**: Scene1 may show *more* slip per contact (all force on one leg → exceeds friction cone), while scene4 distributes force → each contact stays in grip regime. OR: scene2/4 may show *more productive* slip where the robot pushes off multiple contacts.

**Interpretive framework**:
- If scene1 slip fraction >> scene4 → force concentration causes unproductive slip, multi-spoke keeps contacts in grip regime
- If slip fraction is similar but scene4 has higher normal forces per contact → multi-spoke "presses harder" into terrain (weight distributed = each contact lighter, but more contacts = more total friction budget)
- If scene4 shows periodic slip synchronized with drive phase → controlled slip-and-catch gait (biological analogue: cockroach high-speed running)
- Interesting to compare flat vs rough: does rough terrain push more contacts into slip for all morphologies, or selectively?
- **Wheel as limit case**: Wheel spreads force over a larger contact area than spokes (cylindrical mesh vs point-like tips), though simulated as 4 discrete bodies. If wheel has *lower* slip fraction than all spoke designs → smooth geometry distributes force better. If wheel has *higher* slip on rough terrain → the smooth surface slides over asperities instead of gripping them, and discrete spokes that dig into terrain features outperform smooth geometry.

Literature: Full & Koditschek (1999), Goldman et al. (2006) — cockroach slip-and-catch.

### H3. Velocity Ripple (can compute from existing pos_x data)

Multi-spoke designs produce smoother forward velocity within each revolution. Single-spoke concentrates all thrust into one angular window → large velocity oscillation.

**Metric: Velocity ripple** — compute instantaneous vx from pos_x at 2kHz, then:
- Cycle-averaged std: divide time series into one-revolution windows, compute std within each, average across cycles
- Alternatively: FFT of vx, compare amplitude at 1× drive frequency between morphologies

**What we expect**: Scene4 should have lower ripple amplitude at 1× drive freq than scene1, because 4 spokes produce 4 thrust pulses per revolution instead of 1.

**Interpretive framework**:
- Ripple coefficient = intra-cycle std / intra-cycle mean. Scene1 >> scene4 → confirms propulsive discontinuity
- FFT: scene1 should show strong peak at 1× drive freq. Scene4 should show peak at 4× (or spread). The ratio of 1× harmonic amplitude between morphologies quantifies how much smoother propulsion is.
- Compare flat vs rough: if rough terrain *increases* ripple for scene1 but not scene4 → multi-spoke provides terrain noise rejection
- This is distinct from duty factor (H1): duty factor is a contact-level binary, velocity ripple is a kinematic outcome. A robot could have high duty factor but still have high ripple if contacts are unproductive.
- **Wheel as limit case**: Wheel's smooth geometry should produce lower ripple than spoke designs (fewer discrete thrust pulses), though the 4-body discretization may still introduce some periodicity. If scene4 ripple is already near wheel levels → propulsive smoothing saturates by 4 spokes. If wheel ripple *increases* on rough terrain while scene4 stays stable → discrete spokes provide better terrain noise rejection than smooth geometry.

**Note**: Can compute from existing NPZ data (diff(pos_x)/diff(time)). No re-run needed for this analysis alone.

### H4. Phase-Propulsion Correlation (needs drive_angle — requires re-run)

All legs see the same magnetic drive field. With more spokes, the net propulsive force per cycle is a spatial average over more terrain contact points.

**Metric**: Bin instantaneous forward acceleration by drive phase (e.g., 36 bins of 10°). Compute the modulation depth = (max_bin - min_bin) / mean.

**What we expect**: Scene1 has strong modulation (thrust only when the one spoke contacts ground at the right phase). Scene4 has weaker modulation (thrust distributed across 4 phases). On rough terrain, all morphologies get noisier, but scene4 maintains more consistent propulsion.

**Interpretive framework**:
- Modulation depth quantifies "how phase-locked is propulsion to the drive field?"
- Scene1 modulation depth >> scene4 → confirms phase averaging
- If modulation depth is SIMILAR across morphologies → advantage comes from something else (H1/H2), not phase distribution
- Critical comparison: flat vs rough. If rough terrain *destroys* phase structure for scene1 (modulation collapses to noise) but scene4 retains it → "more spokes = better terrain noise rejection" (smoking gun for this hypothesis)
- Could also plot as polar plot: drive angle on θ, mean acceleration on r. Scene1 = elongated ellipse, scene4 = near-circle
- **Wheel as limit case**: Wheel should show weaker phase-locking than spoke designs (smooth geometry reduces angular preference), though the 4-body structure may retain some phase sensitivity. If spoke designs approach wheel's R-squared with increasing spoke count, the trend 1-spoke → 2-spoke → 4-spoke → wheel traces out a phase-averaging curve.

**Note**: drive_angle is NOT in current NPZ (only time, pos_x, pitch). Requires augmented validation re-run.

### H5. Pitch Stability (deprioritized)

We already measure pitch RMS. The additional insight would be whether pitch oscillation *causes* velocity drops (causal coupling) or is just cosmetic.

**What it would tell us**: Pitch RMS says "how much does the robot rock?" Cross-correlation of pitch rate with forward acceleration says "does rocking actually hurt forward motion?" A robot could pitch a lot but still move efficiently if pitching doesn't interfere with thrust generation. Conversely, even small pitch oscillations could kill velocity if they lift contact legs off the ground at critical phases.

**Metric**: Cross-correlation between pitch angular velocity and forward acceleration, with time lag. Peak negative correlation at lag=0 → pitch directly impedes thrust.

**Status**: Low priority. Park unless H1-H4 produce ambiguous results that need pitch as an explanatory variable. Can compute from existing data (pitch + pos_x already in NPZ).

### H6. Contact Position Mapping (exploratory)

Where on the terrain surface are contacts happening? Different morphologies may interact with terrain geometry differently — e.g., single-spoke might get wedged in valleys between asperities while multi-spoke robots ride on top of peaks.

**Metric**: For each leg-terrain contact, record the (x, y, z) contact position. Overlay on terrain heightmap to see spatial distribution.

**What we might see**:
- Multi-spoke robots preferentially contact terrain *peaks* (spoke tips graze high points as they rotate)
- Single-spoke robots spend more time in *valleys* (single leg digs in, gets stuck)
- Contact positions could cluster differently relative to terrain features (slope, local height)
- Height distribution of contact points: scene4 contacts may have higher mean z (riding on top) vs scene1 (digging into valleys)

**Interpretive framework**:
- This is exploratory — we don't have a strong prior on what to expect
- If contact positions differ systematically between morphologies → geometric selection effect is real
- If contact positions are similar → differences come from force/timing, not position
- A scatter plot of contact (x,z) colored by morphology overlaid on terrain cross-section could be visually compelling
- Could also compute "effective ground clearance" = mean contact z - mean terrain z in contact region
- **Wheel as limit case**: Wheel's cylindrical mesh contacts terrain over a broader area than spoke tips, though MuJoCo resolves this as discrete contact points across 4 bodies. Contact position distribution should be smoother/denser than spokes. If wheel contacts cluster at terrain peaks (rides on top) while spoke contacts reach into valleys (dig in), that's a geometric explanation for why discrete spokes grip rough terrain better — they access terrain features that smooth geometry skates over.

**Note**: Requires `leg_contact_pos` from augmented recording.

---

## Implementation Plan

### Step 1: Back up files

```bash
cp milliquad_opt/simulation.py milliquad_opt/simulation.py.bak
cp milliquad_opt/analysis/validate_params.py milliquad_opt/analysis/validate_params.py.bak
```

### Step 2: Add contact recording to simulation.py `_record_state()`

Add `model` parameter to `_record_state()` (needed for `mj_contactForce`).

**New fields per timestep** (fixed-size, per-leg summary):

```python
# For each of 4 legs:
"leg_in_contact":    np.ndarray(4,)     # bool — any geom of this leg touching terrain?
"leg_normal_force":  np.ndarray(4,)     # float — total normal force on this leg (N)
"leg_tangent_force": np.ndarray(4,)     # float — total tangential force magnitude (N)
"leg_contact_pos":   np.ndarray(4, 3)   # float — centroid of contact positions (m)

# Global:
"total_ncon":        int                # total active contacts this step
```

**How to compute**: At each step, loop over `data.contact[:data.ncon]`. For each contact:
1. Use `model.geom_bodyid[contact.geom1]` and `model.geom_bodyid[contact.geom2]` to map geom → body
2. If one body is a leg (body IDs 2-5, i.e. `LEG_BODY_OFFSET` to `LEG_BODY_OFFSET+3`) and the other is world/terrain (body 0), record it
3. Use `mujoco.mj_contactForce(model, data, i, result)` to get the 6D force vector: `[normal, t1, t2, torsional, r1, r2]`
4. Accumulate: `normal_force += result[0]`, `tangent_force += sqrt(result[1]² + result[2]²)`
5. Accumulate contact position centroid

**Key detail**: `mj_contactForce` returns forces in the *contact frame*. `result[0]` is the normal component (always non-negative for active contacts). `result[1:3]` are tangential components.

**Performance**: ~10-20 contacts per step, each `mj_contactForce` call is cheap. Should add <5% overhead to simulation.

### Step 3: Modify validate_params.py to save full raw timeseries

Currently saves: `{rid}_t{trial}_{time,pos_x,pitch}` (3 arrays per trial).

**Complete field list for augmented NPZ** (per trial, all shape (T,) or (T, N)):

| Field suffix | Shape | Source | Needed for |
|---|---|---|---|
| `_time` | (T,) | `traj[i]["time"]` | everything (already saved) |
| `_pos_x` | (T,) | `traj[i]["pos"][0]` | velocity ripple (already saved) |
| `_pos_y` | (T,) | `traj[i]["pos"][1]` | lateral displacement analysis |
| `_pos_z` | (T,) | `traj[i]["pos"][2]` | height variation / terrain interaction |
| `_pitch` | (T,) | `compute_pitch_series()` | pitch stability (already saved) |
| `_vel_x` | (T,) | `traj[i]["vel"][0]` | velocity ripple, phase-propulsion |
| `_vel_y` | (T,) | `traj[i]["vel"][1]` | lateral velocity |
| `_vel_z` | (T,) | `traj[i]["vel"][2]` | vertical bouncing |
| `_joint_pos` | (T, 4) | `traj[i]["joint_pos"]` | gait phase, leg angle analysis |
| `_drive_angle` | (T,) | `traj[i]["drive_angle"]` | phase-propulsion (H4) |
| `_tau_ext` | (T, 4, 3) | `traj[i]["tau_ext"]` | power/COT recomputation |
| `_leg_in_contact` | (T, 4) | NEW from Step 2 | duty factor (H1) |
| `_leg_normal_force` | (T, 4) | NEW from Step 2 | slip fraction (H2) |
| `_leg_tangent_force` | (T, 4) | NEW from Step 2 | slip fraction (H2) |
| `_leg_contact_pos` | (T, 4, 3) | NEW from Step 2 | contact mapping (H6) |
| `_total_ncon` | (T,) | NEW from Step 2 | contact count stats |
| `_body_in_contact` | (T,) | Step 2 v2 | chassis belly-drag detection |
| `_body_normal_force` | (T,) | Step 2 v2 | chassis contact force |
| `_body_tangent_force` | (T,) | Step 2 v2 | chassis friction/drag force |

**Estimated size**: ~53 floats/step × 4000 steps × 8 bytes = ~1.7 MB/trial uncompressed.
- Flat: 80 trials → ~130 MB uncompressed, ~40 MB compressed
- Rough: 120 trials → ~190 MB uncompressed, ~60 MB compressed

**Filename**: Add datetime prefix to avoid overwriting previous results:
```python
ts = datetime.now().strftime("%Y%m%dT%H%M%S")
npz_path = args.run_dir / f"{ts}_validation_trajectories.npz"
csv_path = args.run_dir / f"{ts}_validation_trials.csv"
```

### Step 4: Run validation (user runs manually)

Previous validation commands that produced the mega-figure data:

```bash
cd milliquad_opt

# Flat: 16 refs × 5 trials, select 3 (defaults)
uv run python -m analysis.validate_params results/20260228T013353_rk4_flat --csv

# Step: 12 refs × 5 trials, select 3 (defaults)
uv run python -m analysis.validate_params results/20260228T230022_step_q60_rk-warm --csv

# Rough: 12 refs × 10 trials, select 5
uv run python -m analysis.validate_params results/20260228T202903_rough_spatial_rk4 --csv --n-trials 10 --n-select 5
```

**These exact commands should be re-run** with the augmented simulation.py to produce new NPZ files with contact data. The datetime prefix ensures old files are preserved.

### Step 5: Analysis scripts (all post-processing, no simulation)

#### 5a. `analysis/contact_duty_factor.py` (H1)
- Load NPZ, extract `leg_in_contact` per trial
- Compute: per-morphology duty factor (fraction of timesteps with >= 1 leg in contact)
- Compute: simultaneous contact histogram (0/1/2/3/4 legs in contact)
- Plot: bar chart or time series comparing morphologies
- Compare flat vs rough: does duty factor drop on rough terrain? By how much per morphology?

#### 5b. `analysis/contact_slip_analysis.py` (H2)
- Load NPZ, extract `leg_normal_force` and `leg_tangent_force`
- Compute slip ratio: `F_t / (mu * F_n)` per contact-timestep (where `mu` = fitted sliding friction from config)
- Ratio = 1.0 means at friction cone boundary (slipping)
- Plot: histogram of slip ratio per morphology, overlaid
- Report: fraction of timesteps at >0.95 slip ratio ("near-slip") per morphology

#### 5c. `analysis/velocity_ripple.py` (H3)
- Load NPZ, extract `pos_x` (or `vel_x`) and `time`
- Compute instantaneous vx = diff(pos_x) / diff(time) if using pos_x
- Divide into revolution windows using drive_freq
- Compute intra-revolution std / mean for each window
- Report: mean ripple coefficient per morphology per frequency
- Plot: one-revolution-averaged velocity profile (phase on x-axis, vx on y-axis)

#### 5d. `analysis/phase_propulsion.py` (H4)
- Load NPZ, extract `drive_angle` and `vel_x`
- Compute instantaneous acceleration = diff(vel_x) / diff(time)
- Bin acceleration by drive phase (36 bins)
- Compute modulation depth = (max_bin - min_bin) / mean per morphology
- Plot: phase-averaged acceleration polar plot or line plot
- Compare flat vs rough modulation depth

#### 5e. `analysis/contact_position_map.py` (H6)
- Load NPZ, extract `leg_contact_pos` per trial
- Filter to timesteps where `leg_in_contact` is True
- Overlay contact positions on terrain cross-section (from heightmap PNG, seed=42)
- Plot: scatter of contact (x, z) colored by morphology
- Compute: mean contact height relative to local terrain height
- Compare: do multi-spoke robots ride higher on terrain?

---

## Existing NPZ Contents (current — before augmentation)

Current validation NPZ files contain only 3 arrays per trial:
- `{rid}_t{trial}_time` — shape (4000,)
- `{rid}_t{trial}_pos_x` — shape (4000,)
- `{rid}_t{trial}_pitch` — shape (4000,)

No velocity, no drive_angle, no contact data. **Only H3 (velocity ripple via diff(pos_x)) can run on existing data.** All other hypotheses require the augmented re-run.

---

## Relevant Literature

### Small-Scale Contact Mechanics
- Goldberg et al. (2018) — HAMR insect-scale robot, IEEE RA-L
- Jayaram & Full (2016) — Cockroach crevice traversal, PNAS
- St. Pierre & Bhagavatula (2018-2020) — Sub-gram robots at CMU
- Kim et al. (2013) — Micro Bristle Bot, asymmetric friction locomotion

### Insect Locomotion
- Full & Tu (1991) — 2/4/6-legged cockroach mechanics, J. Exp. Biol.
- Full & Koditschek (1999) — SLIP template, J. Exp. Biol.
- Dickinson et al. (2000) — Integrative animal movement, Science
- Spagna et al. (2007) — Ant locomotion and slip at small scales
- Goldman et al. (2006) — Cockroach dynamic climbing with slip-and-catch

### Multi-Spoke / Whegs
- Quinn et al. (2003) — Whegs concept, IEEE/RSJ IROS
- Saranli et al. (2001) — RHex compliant hexapod, IJRR
- Morrey et al. (2003) — Small quadruped robots, IEEE/RSJ IROS
- Boxerbaum et al. (2005-2012) — Whegs spoke count studies, Case Western

### Rough Terrain at Small Scales
- Li et al. (2015) — Terradynamics, Science
- Sitti (2017) — Mobile Microrobotics, MIT Press
- Li et al. (2019) — Cockroach body+leg mechanics on rough terrain, J. Exp. Biol.
- Zarrouk et al. (2013-2018) — Minimalist spoke-based locomotion

### Slip and Contact Modes
- Aguilar et al. (2016) — Locomotion robophysics review, Rep. Prog. Phys.
- Zhang et al. (2013) — Ground fluidization for lightweight robots, IJRR
- Qian & Goldman (2015) — Walking on yielding ground, RSC Advances
- Garcia et al. (2000) — Passive compass gait stability

### Magnetic Actuation
- Hu et al. (2018) — Multimodal magnetic robot, Nature
- Ren et al. (2019) — Magnetic multi-appendage locomotion, Nature Comms
- Diller & Sitti (2014) — Micro-scale mobile robotics review
- Huang et al. (2019) — Programmable magnetic micro-robot gaits, Nature Comms

---

## Progress Log

### Step 1: Back up files — DONE (2026-03-02)
- `simulation.py.bak` and `validate_params.py.bak` created

### Step 2: Add contact recording to simulation.py — DONE (2026-03-02)
- Added `_extract_contact_data(model, data)` helper (~50 lines)
- Updated `_record_state` signature: `(trajectory, data, step_cache)` → `(trajectory, model, data, step_cache)`
- Fixed monkey-patch in `energy_analysis.py` to match new signature
- New fields per timestep: `leg_in_contact` (4,), `leg_normal_force` (4,), `leg_tangent_force` (4,), `leg_contact_pos` (4,3), `total_ncon` (int)

### Step 3: Augment validate_params.py — DONE (2026-03-02)
- Added `_store_trajectory_arrays()` helper with float32 casting (16 field types per trial)
- Added datetime prefix to output filenames (prevents overwriting old data)
- Float32 casting: 2× size reduction vs float64, max relative error ~6e-8 (f32 machine epsilon)

### Step 4: Run validation — DONE (2026-03-02, re-run with body contact)

19 fields/trial (16 original + 3 body contact). Previous 16-field NPZ files were verified identical on shared arrays, then deleted.

**Current NPZ files** (19 fields):
- Flat: `20260302T141425_validation_trajectories.npz` (53 MB, 80 trials × 19 fields)
- Step: `20260302T141529_validation_trajectories.npz` (72 MB, 60 trials × 19 fields)
- Rough: `20260302T141807_validation_trajectories.npz` (49 MB, 120 trials × 19 fields)

**Re-run commands** (from `milliquad_opt/`):
```bash
uv run python -m analysis.validate_params results/20260228T013353_rk4_flat --csv
uv run python -m analysis.validate_params results/20260228T230022_step_q60_rk-warm --csv
uv run python -m analysis.validate_params results/20260228T202903_rough_spatial_rk4 --csv --n-trials 10 --n-select 5
```

**Parity note** (from first run): All shared fields (time, pos_x, pitch) matched old NPZ files within f32 tolerance across all three terrains (240/240 flat, 180/180 step, 360/360 rough).

### Step 5: Analysis scripts — DONE (2026-03-02)

All scripts in `analysis/l2_l4/` subfolder (self-contained, do NOT touch `_common.py` or existing plotting pipeline).

**Shared module**: `_trial_filter.py` — consistent trial validation and spatial gating across all scripts.
- `is_valid_trial()`: gate-clearing (GATE_END) + inversion check (pitch std > 30°)
- `active_mask()`: flat = `time >= 0.1s`, rough = time + spatial `[0.005, 0.14)`, step = spatial `[0.05, 0.096)`
- Constants mirror `plot_validation.py` (GATE_END, GATE_EXEMPT, INVERTED_PITCH_THRESHOLD)

**Scripts** (all support multi-dir cross-terrain comparison):

| Script | Hypothesis | Key metric | Status |
|---|---|---|---|
| `contact_duty_factor.py` | H1 | Duty factor, simultaneous contact histogram | Tested flat/step/rough |
| `contact_slip_analysis.py` | H2 | Slip fraction (F_t / mu·F_n >= 0.95) | Tested flat/rough |
| `velocity_ripple.py` | H3 | Ripple coefficient, FFT amplitude at 1× drive freq | Tested flat/rough |
| `phase_propulsion.py` | H4 | Phase R² (var(bin_means)/var(ax)), amplitude, peak phase | Tested flat/rough |
| `contact_position_map.py` | H6 | Mean contact z/y position, spatial distribution | Tested flat/rough |

**Usage** (from `milliquad_opt/`):
```bash
uv run python -m analysis.l2_l4.contact_duty_factor results/20260228T013353_rk4_flat results/20260228T202903_rough_spatial_rk4
```

---

## Lessons Learned

1. **Float32 is sufficient for all physical quantities in this simulation.** Max relative error from f64→f32 cast is ~6e-8 across all fields (time, position, pitch, velocity, torque, contact forces). Saves 2× storage with zero practical accuracy loss.

2. **HDF5 shuffle+gzip adds negligible benefit over NPZ for this data.** Benchmarked NPZ f64, NPZ f32, HDF5 gzip f64/f32/f32+gzip9. Float32 cast dominates the size reduction (~2×). HDF5 shuffle on top of f32 gives only ~3% additional compression. Not worth the h5py dependency.

3. **Datetime-prefixed filenames prevent accidental data loss.** Old NPZ/CSV files are preserved alongside new ones. Critical when iterating on recording format — can always diff old vs new.

4. **`_record_state` signature change propagates to monkey-patches.** `energy_analysis.py` patches `_record_state` to add energy recording. Adding `model` parameter broke it silently until tested. Any future signature changes need grep for `_record_state` across codebase.

5. **Contact force API**: `mujoco.mj_contactForce(model, data, i, result)` returns 6D vector in contact frame. `result[0]` = normal (≥0), `result[1:3]` = tangential. Must map geom→body via `model.geom_bodyid` to identify which leg is involved.

6. **Trial filtering is essential for terrain analysis.** Without gate-clearing and spatial gating, rough terrain metrics are contaminated by post-terrain acceleration on flat ground, and step terrain by cliff-fall artifacts. The existing plotting pipeline (plot_validation.py) had already solved this — reuse the same constants.

7. **Phase-propulsion modulation depth is a broken metric.** `(max_bin - min_bin) / mean_accel` explodes at steady state because mean acceleration ≈ 0. Phase R² = `var(bin_means) / var(all_ax)` is bounded [0,1] and directly interpretable as "fraction of acceleration variance explained by drive phase."

8. **Keep analysis scripts self-contained from existing pipelines.** `_common.py` controls plotting infrastructure — modifying it for prototype analysis risks breaking production plots. Created `analysis/l2_l4/_trial_filter.py` as a local shared module instead.

9. **Body contact (belly-drag) is a dominant failure mode for L1.** L1 spends 20–60% of active time with chassis on the ground on step/rough terrain. Adding `body_in_contact` to the recording was essential — without it, L1's high zero-leg-contact fraction looks like "legs in the air" when it's actually "belly on the ground." Different morphologies fail differently: L1 belly-drags, WR gets stuck with legs gripping but no forward progress.

10. **Wheel model shares leg topology with spoke robots.** The wheel MJCF (`robot_wheel.xml`) has 4 hinge joints (FR/FL/BR/BL) and 4 bodies at IDs 2-5, same as spoke models. `_extract_contact_data` maps these as "legs" — so wheel's per-leg arrays represent per-wheel-segment data. No code changes needed, but interpret wheel's `leg_in_contact`, force, and position arrays as segment-level, not limb-level. H3 and H4 (velocity ripple, phase-propulsion) don't use contact arrays and are unaffected.

---

## Step 6: Plotting & Visualization Plan

### Data Inventory

| Terrain | Morphologies | Frequencies | Trials/combo | Total trials |
|---|---|---|---|---|
| Flat | 1-spoke, 2-spoke, 4-spoke, wheel | 10, 20, 30, 50 Hz | 5 | 80 |
| Step | 1-spoke, 2-spoke, 4-spoke, wheel | 10, 20, 30 Hz | 5 | 60 |
| Rough | 1-spoke, 2-spoke, 4-spoke, wheel | 10, 30, 50 Hz | 10 | 120 |

**16 fields per trial**: time, pos_x/y/z, vel_x/y/z, pitch, drive_angle, joint_pos(4), leg_in_contact(4), leg_normal_force(4), leg_tangent_force(4), leg_contact_pos(4,3), total_ncon, tau_ext(4,3)

### Raw Plots per Hypothesis

#### H1: Contact Redundancy / Duty Factor
**Fields**: `leg_in_contact` (T, 4) bool

| # | Plot | What it shows | Axes |
|---|---|---|---|
| 1a | Contact raster | Per-leg on/off timeline for one trial | x=time, y=leg_id, color=in_contact |
| 1b | Simultaneous contact histogram | Distribution of 0/1/2/3/4 legs in contact | x=n_legs, y=fraction, grouped by morphology |
| 1c | Duty factor by freq | Duty factor (>=1 leg) per morphology across frequencies | x=freq, y=duty_factor, lines=morphology |
| 1d | Contact count timeseries | How many legs in contact over time | x=time, y=count(0-4), one line per morphology |

#### H2: Slip vs Grip
**Fields**: `leg_normal_force`, `leg_tangent_force`, `leg_in_contact`

| # | Plot | What it shows | Axes |
|---|---|---|---|
| 2a | Slip ratio histogram | Distribution of F_t/(mu*F_n) across all contacts | x=ratio(0-1+), y=density, one curve per morphology |
| 2b | F_t vs F_n scatter | Each contact as a point, with friction cone line | x=F_n, y=F_t, color=morphology, line=mu*F_n |
| 2c | Slip fraction by freq | Fraction of contacts at >0.95 slip | x=freq, y=slip_frac, lines=morphology |
| 2d | Normal force distribution | How hard each leg pushes | x=F_n, y=density, per morphology |

#### H3: Velocity Ripple
**Fields**: `vel_x`, `drive_angle`, `time`

| # | Plot | What it shows | Axes |
|---|---|---|---|
| 3a | Phase-folded velocity | vel_x overlaid on one drive revolution | x=drive_phase(0-2pi), y=vel_x, per morphology |
| 3b | FFT power spectrum | Harmonics of drive frequency in velocity | x=freq_ratio(1x,2x,4x), y=amplitude, per morphology |
| 3c | Ripple coefficient by freq | Intra-cycle std/mean across drive freqs | x=freq, y=ripple, lines=morphology |
| 3d | Raw velocity timeseries | vel_x over time showing oscillation pattern | x=time, y=vel_x, one representative trial per morph |

#### H4: Phase-Propulsion Correlation
**Fields**: `vel_x`, `drive_angle`, `time`

| # | Plot | What it shows | Axes |
|---|---|---|---|
| 4a | Phase-averaged acceleration | Mean ax in 10-deg drive phase bins | x=phase(0-360), y=mean_ax, per morphology |
| 4b | Polar propulsion plot | Same as 4a but on polar axes | theta=phase, r=ax, per morphology |
| 4c | Phase R-squared by freq | How much acceleration is phase-locked | x=freq, y=R2, lines=morphology |
| 4d | Phase-acceleration heatmap | 2D density of (phase, ax) per morphology | x=phase, y=ax, color=density |

#### H6: Contact Position Mapping
**Fields**: `leg_contact_pos` (T,4,3), `leg_in_contact`

| # | Plot | What it shows | Axes |
|---|---|---|---|
| 6a | Contact scatter (x,z) | Where on terrain legs touch | x=pos_x, y=pos_z, color=morphology |
| 6b | Contact height histogram | Distribution of contact z | x=z_mm, y=density, per morphology |
| 6c | Contact y spread | Lateral distribution of contacts | x=pos_y, y=density, per morphology |

### Summary / Comparative Analysis (cross-hypothesis)

| # | Analysis | Question it answers |
|---|---|---|
| S1 | Terrain degradation matrix | For each metric x morphology: (rough - flat) / flat. Who degrades least? |
| S2 | Morphology scorecard | One row per morphology, columns = all metrics. Normalized 0-1 (best=1). Radar/spider chart possible. |
| S3 | Duty factor vs velocity | Does higher duty factor → higher velocity? Scatter across all (morph, freq, terrain) combos. |
| S4 | Slip fraction vs ripple | Does more slip → more velocity oscillation? Correlation across conditions. |
| S5 | Phase R-squared vs duty factor | Is phase-locked propulsion related to contact regularity? |
| S6 | Freq scaling | How does each metric scale with drive frequency? All metrics on one freq-axis figure, faceted by morphology. |
| S7 | Step vs rough comparison | Same metrics on step terrain — does morphology ranking change between obstacle types? |

---

## Step 7: Hypothesis Plotting — STARTED (2026-03-02)

### Folder structure

Plotting scripts organized by hypothesis under `analysis/l2_l4/H{n}/`, with timestamped outputs in `figures/` subfolders. Shared style in `analysis/l2_l4/_plot_style.py`.

```
analysis/l2_l4/
  _plot_style.py              # MORPH_COLORS, MORPH_LABELS, rcParams (new)
  _trial_filter.py            # trial validation + gating (existing)
  contact_duty_factor.py      # text-output analysis (existing)
  H1/
    __init__.py
    plot_contact_histogram.py  # 1b — grouped bar chart, flat/step/rough
    plot_contact_raster.py     # 1a — per-leg on/off raster, sanity check
    figures/                   # timestamped PNGs
```

### H1 plots — DONE (2026-03-02)

**1b: Simultaneous contact histogram** (`plot_contact_histogram.py`)
- Grouped bars: x = 0–4 simultaneous leg contacts + "Body" bin (chassis-terrain contact fraction), y = fraction of timesteps, grouped by morphology
- Grid: rows = terrain, columns = drive frequency. Unified freq grid across terrains (blank cells where freq not tested).
- Per-trial mean ± std error bars. Annotated with N=valid/total per morphology.
- `--failed` flag inverts trial selection (shows failing trials instead of passing). Uses time-only mask since failed robots may not reach terrain region. Terrains with zero failures (e.g. flat) are omitted automatically.
- Body bin is hatched, separated by dashed line from leg bins.
- Old iterations archived in `figures/archive/`.

**Key findings from H1**:
- All morphologies peak at 0 leg contacts across all terrains — legs are airborne most of the time.
- L1 has ~60–70% zero-contact fraction. L4 and WR distribute more evenly across 1–3 contacts.
- **Body contact is the big reveal**: L1 shows ~20–60% body contact on step/rough terrain — the robot is belly-dragging. L4 and WR keep the body elevated.
- Contact redundancy isn't just "more legs touch ground" — it's "more legs keep the body OFF the ground."
- **Failure mode divergence**: L1 fails by belly-sliding with no leg contact. WR fails by having leg contact but no forward progress (stuck on terrain features). L2/L4 rarely fail.
- **WR pass rates are very low**: step 3–5/15, rough 7/30. WR's theoretical continuous-contact advantage doesn't translate to traversal success.
- Frequency matters: at 50 Hz, L4 zero-contact fraction approaches L1 levels on flat terrain. On rough, high frequency helps WR pass rate (4/10 at 50 Hz vs 0/10 at 10 Hz).

**1a: Contact raster** (`plot_contact_raster.py`)
- Per-leg on/off heatmap for one representative trial (median duty factor) per morphology.
- Picks highest available frequency by default, configurable with `--freq`.
- Sanity check — confirms histogram patterns visually.

**Usage** (from `milliquad_opt/`):
```bash
# 1b — passing trials, all terrains
uv run python -m analysis.l2_l4.H1.plot_contact_histogram \
    results/20260228T013353_rk4_flat \
    results/20260228T230022_step_q60_rk-warm \
    results/20260228T202903_rough_spatial_rk4

# 1b — failed trials
uv run python -m analysis.l2_l4.H1.plot_contact_histogram \
    results/20260228T013353_rk4_flat \
    results/20260228T230022_step_q60_rk-warm \
    results/20260228T202903_rough_spatial_rk4 --failed

# 1a — single terrain, specific freq
uv run python -m analysis.l2_l4.H1.plot_contact_raster \
    results/20260228T202903_rough_spatial_rk4 --freq 30
```

---

## Notes

- Literature citations are from training knowledge (through May 2025) — verify exact titles/years via Google Scholar before manuscript use
- Rough terrain is deterministic (seed=42), so re-running with augmented recording reproduces identical terrain
- All analysis scripts are pure post-processing on NPZ files — no simulation dependency after Step 4
- The augmented NPZ captures ALL data needed for H1-H6 so we only re-run validation once
