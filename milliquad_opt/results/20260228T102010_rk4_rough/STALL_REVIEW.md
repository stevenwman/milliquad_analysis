# Rough Terrain Stall Review

Visual inspection of `trajectory_overview.png` and video recordings.

## Method
- `min_window_velocity` (5-period sliding window of mean |vx|) flags candidates
- Visual review of x(t) trajectories determines ground truth
- No single mwv threshold cleanly separates stuck from slow-but-moving

## Selected trials — stall verdicts

| ref_id | trial | vx (mm/s) | COT | mwv (mm/s) | verdict |
|--------|-------|-----------|-----|------------|---------|
| scene1_f50 | t4 | 10.7 | 50.7 | 4.5 | stuck |
| scene4_f30 | t0 | 127.1 | 2.5 | 6.7 | **keep** |
| scene4_f30 | t1 | 111.5 | 2.9 | 11.4 | **keep** |
| scene4_f30 | t4 | 149.6 | 2.3 | 6.4 | **keep** |
| scene4_f50 | t4 | 40.7 | 13.1 | 6.0 | stuck |
| scene_wheel_f10 | t0 | 30.3 | 2.5 | 15.5 | stuck |
| scene_wheel_f10 | t2 | 37.6 | 1.7 | 1.8 | stuck |
| scene_wheel_f10 | t3 | 11.7 | 5.7 | 1.7 | stuck |
| scene_wheel_f30 | t0 | 10.3 | 25.0 | 2.0 | stuck |
| scene_wheel_f30 | t3 | 101.4 | 3.4 | 7.8 | **keep** |
| scene_wheel_f30 | t4 | 43.0 | 8.4 | 6.2 | stuck |
| scene_wheel_f50 | t0 | 17.3 | 48.5 | 7.5 | stuck |
| scene_wheel_f50 | t1 | 25.1 | 32.6 | 7.4 | stuck |
| scene_wheel_f50 | t2 | 10.3 | 70.4 | 2.5 | stuck |

All other selected trials: **keep**.
