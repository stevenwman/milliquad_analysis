# MuJoCo Gotchas & Lessons Learned

## Arena Memory (`<size memory>`)

The MuJoCo arena (`mjData.narena`) holds both constraint solver data and the computation stack. Two things can blow it up:

1. **Heightfield contacts**: Dense heightfields generate many contact candidates. Even with few final contacts (`ncon`), the collision broadphase uses arena space.

2. **Noslip solver** (`noslip_iterations > 0`): Allocates a **dense nefc x nefc matrix** in the arena. Memory scales as `nefc^2 * 8 bytes`. Examples:
   - nefc=480 (48 contacts) → 1.8 MB
   - nefc=1320 (132 contacts) → 14 MB

   If the arena is too small, `mj_step` throws `mj_stackAlloc: out of memory`. The `max` field in the error is the **stack portion** of the arena (total minus constraint data), NOT the total arena.

**Fix**: Set `<size memory="32M"/>` in generated terrain XMLs. Default auto-allocation (~66KB for these robots) is far too small when noslip is active on heightfield terrain.

**Gotcha**: `model.narena` can report the correct value (e.g. 2MB) at load time, but the noslip solver still overflows because it consumes the arena at simulation time, leaving only a few KB for the stack. The symptom is `max = 7712` with 2MB arena — the dense matrix ate everything.

### Debugging arena issues

- Add `print(f"narena={model.narena}")` after `MjModel.from_xml_path()` to verify the XML is respected
- If `narena` is correct but the crash still says small `max`, the problem is runtime arena consumption (noslip solver, many contacts), not the XML
- `<size memory>` uses human-readable suffixes: `K`, `M`, `G`
- `<size memory>` and `nstack` are mutually exclusive — don't set both

### Cost of large arenas in multiprocessing

Each worker allocates its own arena. 20 workers × 32MB = 640MB. Acceptable on modern machines, but worth noting if scaling to 100+ workers.

## Terrain MJCF Editing

### Write edited XMLs alongside originals

When programmatically editing MJCF files (e.g., adding heightfield terrain), the modified XML must be written **in the same directory** as the original. MuJoCo resolves `<include>` and mesh paths relative to the XML file's location. Writing to `/tmp/` or elsewhere breaks all relative references.

Convention: `scene_X_terrain_tmp.xml` alongside `scene_X.xml`. Clean up in `try/finally`.

### Heightfield `z_bottom` must be > 0

`generate_terrain_hfield()` can return `data_min = 0.0` for flat-bottomed heightfields. MuJoCo silently misbehaves if `z_bottom = 0`. Clamp to `max(z_bottom, 0.001)`.

## Contact & Friction Override

### Global friction application via `mjENBL_OVERRIDE`

To apply fitted friction parameters to ALL geoms (including dynamically-generated terrain):

```python
model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_OVERRIDE
model.geom_condim[:] = 6  # torsional + rolling friction for all geoms
```

This overrides per-geom condim settings. No need to manually set friction on terrain geoms — the override handles it. `condim=6` enables the full 5-element friction cone (sliding + torsional + rolling).

## Rough Terrain Specifics

### `rough_mild` (0.5mm std) barely affects robots

At the milliquad scale (~6mm), 0.5mm roughness amplitude is too small to meaningfully perturb locomotion. Use `rough_harsh` (1mm std) for differentiation between morphologies.

### Step terrain cliff-fall artifact

Robots that reach the end of a staircase fall off the final step. This produces:
- yaw ~180° (robot spins on impact)
- large tumble/lateral drift in the last few timesteps

Mitigations:
- Increase `final_step_length` so robots don't reach the edge
- Set yaw weight = 0 in step cost function
- Use BEST-trial aggregation (not median) to tolerate 1-2 corrupted trials
- Spatial gating: stop measuring before the cliff
