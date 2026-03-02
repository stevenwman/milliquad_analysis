# Why Scene2 (2-spoke) and Scene4 (4-spoke) Do Well on Rough Terrain

## Background

The "scene #" refers to the number of spokes on a leg connected to the hip joint:
- **Scene1**: Single leg (standard single-spoke configuration)
- **Scene2**: 2 legs 180° apart (2-spoke)
- **Scene4**: 4 spokes 90° apart (4-spoke)
- **Wheel**: Continuous cylindrical contact

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

**Estimated size**: ~50 floats/step × 4000 steps × 8 bytes = 1.6 MB/trial uncompressed.
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

## Notes

- Literature citations are from training knowledge (through May 2025) — verify exact titles/years via Google Scholar before manuscript use
- Rough terrain is deterministic (seed=42), so re-running with augmented recording reproduces identical terrain
- H3 (velocity ripple) can run on existing NPZ data (diff(pos_x)). All other hypotheses require re-run.
- H4 (phase-propulsion) needs `drive_angle` which is NOT in current NPZ.
- All analysis scripts are pure post-processing on NPZ files — no simulation dependency after Step 4.
- The augmented NPZ is designed to capture ALL data needed for H1-H6 so we only re-run validation once.
