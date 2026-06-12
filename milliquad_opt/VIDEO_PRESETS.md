# Video Recording Presets

## Side-view close-up (40x slow-mo, 30fps output)

```bash
# Flat terrain
uv run python validate_single.py results/<flat_run_dir> \
  --ref-id <scene>_f<freq> --record \
  --cam-lookat 0.07 0 0 --cam-elevation -5 --cam-distance 0.1 \
  --slow-mo 40 --no-tracking --duration 1.0

# Step terrain
uv run python validate_single.py results/<step_run_dir> \
  --ref-id <scene>_f<freq> --record \
  --cam-lookat 0.07 0 0 --cam-elevation -5 --cam-distance 0.1 \
  --slow-mo 40 --no-tracking --duration 1.0
```

### Recent run directories
- Flat: `20260303T192801_flat_tg`
- Step: `20260303T151416_step_065gate`
- Rough: `20260303T224229_rough_tg`

### Example ref-ids
- `scene1_f10`, `scene2_f30`, `scene4_f30`, `scene_wheel_f10`, `scene_wheel_f30`

### Notes
- `--slow-mo N` alone sets capture_fps = 30 * N (output is always 30fps)
- Max meaningful slow-mo is ~66x (sim timestep = 2000 Hz, 2000/30 = 66)
- `--no-tracking` = fixed camera (no robot tracking)
- `--duration` = sim time in seconds
