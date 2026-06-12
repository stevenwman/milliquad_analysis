# Plan: Make terrain XMLs device-agnostic

## Context

`generate_terrain_xmls.py` writes absolute paths into rough terrain XMLs for the heightfield PNG reference (e.g., `file="/home/sman/.../scene_1_rough.png"`). When a different machine clones the repo, these paths are wrong — forcing the user to re-run `generate_terrain_xmls.py`, which creates dirty working tree diffs and merge headaches across devices.

Step XMLs don't have this problem (no external file references). Flat XMLs use relative `<include file="robot_1.xml"/>` and work fine.

## Fix

**One-line change in `generate_terrain_xmls.py`**: use a relative PNG path with `../` prefix.

- **File**: `milliquad_opt/generate_terrain_xmls.py`, line 163
- **Before**: `hf.set("file", png_abs)` → `/home/sman/.../scene_1_rough.png`
- **After**: `hf.set("file", f"../{png_path.name}")` → `../scene_1_rough.png`

Also remove the now-unused `png_abs` variable (line 144).

Then regenerate all 4 rough XMLs and commit. After this, no device ever needs to re-run `generate_terrain_xmls.py` — the tracked XMLs work everywhere.

### Why `../` and not just the filename?

Toy test revealed that `file="scene_1_rough.png"` (filename only) **fails**:

```
ValueError: Error: could not open file 'assets/scene_1_rough.png'
```

**Root cause**: The rough XML uses `<include file="robot_1.xml"/>`, and `robot_1.xml` sets `meshdir="assets"`. MuJoCo's `meshdir` applies globally to ALL `file=` attributes resolved after that point — including the hfield PNG. So MuJoCo looks for `assets/scene_1_rough.png` instead of `scene_1_rough.png`.

The fix is `../scene_1_rough.png` which resolves to `assets/../scene_1_rough.png` = the XML's own directory. Verified working in toy test.

**Note**: `<include file="robot_1.xml"/>` itself works with just the filename because `<include>` is resolved before `meshdir` takes effect — it uses the XML's directory, not `meshdir`.

## Files modified

1. `milliquad_opt/generate_terrain_xmls.py` — use relative PNG path
2. `milliquad_opt/robots/quad/scene_1_rough.xml` — regenerated
3. `milliquad_opt/robots/quad/scene_2_rough.xml` — regenerated
4. `milliquad_opt/robots/quad/scene_4_rough.xml` — regenerated
5. `milliquad_opt/robots/wheel/scene_wheel_rough.xml` — regenerated

## Toy Test Results

```
# Test 1: filename only → FAIL
file="scene_1_rough.png"
→ ValueError: could not open file 'assets/scene_1_rough.png'

# Test 2: ../ prefix → PASS
file="../scene_1_rough.png"
→ OK (MuJoCo loaded model successfully)
```

## Verification (after applying fix)

```bash
cd milliquad_opt
uv run python generate_terrain_xmls.py
uv run python -c "import mujoco; m=mujoco.MjModel.from_xml_path('robots/quad/scene_1_rough.xml'); print('OK')"

# Verify no absolute paths remain
grep -r 'file="/' robots/
```

## Note

The running optimizer loads XMLs from disk on every sim call (no caching). Changing XMLs mid-run is technically safe if the resolved file is identical, but better to wait until the run finishes.
