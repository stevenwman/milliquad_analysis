"""One-time script: generate step and rough terrain XMLs from flat scene XMLs.

Reads flat scene XMLs from robots/quad/ and robots/wheel/, injects terrain
geometry, and writes terrain variants alongside the originals.

Step terrain: 8 box geoms (1mm high, 4.5mm long, 50mm flat lead, 20mm platform)
Rough terrain: heightmap hfield (seed=42, 1mm std), tiled 3× along X, walls

All generated XMLs inherit integrator="RK4" from the flat base.

Usage:
    uv run python generate_terrain_xmls.py
"""

import pathlib
import sys
import xml.etree.ElementTree as ET

import imageio.v3 as iio
import numpy as np

# Add project root so we can import utils/terrain_mesh.py
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.terrain_mesh import generate_heightmap

# ---------------------------------------------------------------------------
# Flat scene XMLs (inputs)
# ---------------------------------------------------------------------------
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

FLAT_SCENES = {
    "scene1": SCRIPT_DIR / "robots" / "quad" / "scene_1_flat.xml",
    "scene2": SCRIPT_DIR / "robots" / "quad" / "scene_2_flat.xml",
    "scene4": SCRIPT_DIR / "robots" / "quad" / "scene_4_flat.xml",
    "scene_wheel": SCRIPT_DIR / "robots" / "wheel" / "scene_wheel_flat.xml",
}

# ---------------------------------------------------------------------------
# Step terrain geometry (from mujoco_refactor/config_step.py)
# ---------------------------------------------------------------------------
STEP_PRESET = {
    "step_height": 0.001,        # 1mm
    "step_length": 0.0045,       # 4.5mm
    "step_count": 8,
    "final_step_length": 0.02,   # 20mm extended platform
    "step_width": 0.1,           # 100mm
    "flat_lead": 0.05,           # 50mm flat ground before first step
}

# ---------------------------------------------------------------------------
# Rough terrain geometry (from mujoco_refactor/config_rough.py)
# ---------------------------------------------------------------------------
TERRAIN_NX = 10
TERRAIN_NY = 6
TERRAIN_SL = 0.005       # 5mm tile side
TERRAIN_HEIGHT_MEAN = 0.002
TERRAIN_HEIGHT_STD = 0.001
TERRAIN_Z_SAFE = 0.00025
TERRAIN_SEED = 42
N_TILES = 3
FLAT_LEAD = 0.005        # 5mm flat lead
PIXELS_PER_SQUARE = 8


# ---------------------------------------------------------------------------
# Step XML generation
# ---------------------------------------------------------------------------

def inject_steps(flat_xml: pathlib.Path, out_xml: pathlib.Path) -> None:
    """Add step box geoms to a flat scene XML and write to out_xml."""
    tree = ET.parse(flat_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    flat_lead = STEP_PRESET["flat_lead"]
    step_height = STEP_PRESET["step_height"]
    step_length = STEP_PRESET["step_length"]
    step_count = STEP_PRESET["step_count"]
    final_step_length = STEP_PRESET["final_step_length"]
    step_width = STEP_PRESET["step_width"]

    # Arena memory for noslip solver with extra step contacts
    size_elem = root.find("size")
    if size_elem is None:
        size_elem = ET.SubElement(root, "size")
    size_elem.set("memory", "128M")

    for i in range(step_count):
        is_final = (i == step_count - 1)
        length = final_step_length if is_final else step_length

        if is_final:
            pos_x = flat_lead + (step_count - 1) * step_length + length / 2.0
        else:
            pos_x = flat_lead + i * step_length + length / 2.0
        pos_z = (i + 1) * step_height - step_height / 2.0

        geom = ET.SubElement(worldbody, "geom")
        geom.set("name", f"step_{i}")
        geom.set("type", "box")
        geom.set("size", f"{length/2.0} {step_width/2.0} {step_height/2.0}")
        geom.set("pos", f"{pos_x} 0.0 {pos_z}")
        geom.set("rgba", "0.5 0.5 0.5 1")

    tree.write(str(out_xml))
    print(f"  step  → {out_xml.name}")


# ---------------------------------------------------------------------------
# Rough XML generation
# ---------------------------------------------------------------------------

def inject_rough(flat_xml: pathlib.Path, out_xml: pathlib.Path) -> None:
    """Generate heightfield PNG and inject into a flat scene XML."""
    logical_heights = generate_heightmap(
        nX=TERRAIN_NX, nY=TERRAIN_NY,
        height_mean=TERRAIN_HEIGHT_MEAN, height_std=TERRAIN_HEIGHT_STD,
        z_safe=TERRAIN_Z_SAFE, seed=TERRAIN_SEED,
    )

    # Transpose to (NY, NX) — MuJoCo hfield: rows=Y, columns=X
    logical_heights = logical_heights.T

    # Tile N_TILES times along X (axis=1 = columns)
    tiled_heights = np.tile(logical_heights, (1, N_TILES))

    # Upsample for blocky appearance
    hires = np.kron(tiled_heights, np.ones((PIXELS_PER_SQUARE, PIXELS_PER_SQUARE)))

    total_nx = TERRAIN_NX * N_TILES
    x_half = total_nx * TERRAIN_SL / 2.0
    y_half = TERRAIN_NY * TERRAIN_SL / 2.0
    z_top = float(tiled_heights.max())
    z_bottom = 0.001

    # Normalize to [0, 1] and write PNG
    normalized = np.clip(hires / z_top, 0.0, 1.0)
    img = (normalized * np.iinfo(np.uint16).max).astype(np.uint16)

    png_path = out_xml.with_suffix(".png")
    iio.imwrite(str(png_path), img)
    png_abs = str(png_path.resolve())

    # Parse flat MJCF and inject hfield
    tree = ET.parse(flat_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    # Arena memory for noslip solver with heightfield contacts
    size_elem = root.find("size")
    if size_elem is None:
        size_elem = ET.SubElement(root, "size")
    size_elem.set("memory", "128M")

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    hf = ET.SubElement(asset, "hfield")
    hf.set("name", "rough_terrain")
    hf.set("file", png_abs)
    hf.set("size", f"{x_half} {y_half} {z_top} {z_bottom}")

    pos_x = FLAT_LEAD + x_half
    geom = ET.SubElement(worldbody, "geom")
    geom.set("name", "rough_terrain_geom")
    geom.set("type", "hfield")
    geom.set("hfield", "rough_terrain")
    geom.set("pos", f"{pos_x} 0.0 0.0")
    geom.set("rgba", "0.6 0.55 0.5 1")

    # Side walls
    wall_height = 0.01
    wall_thick = 0.001
    for side, y_sign in [("left", 1), ("right", -1)]:
        wall_y = y_sign * (y_half + wall_thick)
        wall = ET.SubElement(worldbody, "geom")
        wall.set("name", f"wall_{side}")
        wall.set("type", "box")
        wall.set("size", f"{x_half} {wall_thick} {wall_height}")
        wall.set("pos", f"{pos_x} {wall_y} {wall_height}")
        wall.set("rgba", "0.7 0.8 1.0 0.3")
        wall.set("contype", "1")
        wall.set("conaffinity", "1")

    # Back wall to prevent robot escaping behind the terrain
    back_x = FLAT_LEAD  # at terrain start edge
    back_wall = ET.SubElement(worldbody, "geom")
    back_wall.set("name", "wall_back")
    back_wall.set("type", "box")
    back_wall.set("size", f"{wall_thick} {y_half + 2 * wall_thick} {wall_height}")
    back_wall.set("pos", f"{back_x} 0.0 {wall_height}")
    back_wall.set("rgba", "0.7 0.8 1.0 0.3")
    back_wall.set("contype", "1")
    back_wall.set("conaffinity", "1")

    tree.write(str(out_xml))
    print(f"  rough → {out_xml.name} + {png_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Generating terrain XMLs...\n")

    for name, flat_path in FLAT_SCENES.items():
        if not flat_path.exists():
            print(f"  SKIP {name}: {flat_path} not found")
            continue

        print(f"{name}:")
        out_dir = flat_path.parent

        # Derive output names from flat name (e.g. scene_1_flat.xml → scene_1_step.xml)
        stem = flat_path.stem.replace("_flat", "")

        step_xml = out_dir / f"{stem}_step.xml"
        rough_xml = out_dir / f"{stem}_rough.xml"

        inject_steps(flat_path, step_xml)
        inject_rough(flat_path, rough_xml)
        print()

    print("Done. Verify with: uv run python -c \"import mujoco; m=mujoco.MjModel.from_xml_path('robots/quad/scene_1_step.xml'); print('OK', m.opt.integrator)\"")


if __name__ == "__main__":
    main()
