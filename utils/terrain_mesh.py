"""Generate rough terrain from a random heightmap.

Creates a rectangular grid of nX * nY square tiles (side length sL) with
white-noise heights.  The entire block is shifted so the shortest tile still
has z_safe material between its top and the flat Z=0 base.

Two output formats:
  - OBJ mesh  (``generate_terrain_obj``)
  - MuJoCo hfield PNG + size tuple  (``generate_terrain_hfield``)
Both share the same heightmap generator.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np


# ---------------------------------------------------------------------------
# Heightmap generation (shared)
# ---------------------------------------------------------------------------

def generate_heightmap(
    nX: int,
    nY: int,
    height_mean: float,
    height_std: float,
    z_safe: float,
    seed: int | None = None,
) -> np.ndarray:
    """Sample a white-noise heightmap and shift so min tile = z_safe.

    Returns
    -------
    heights : np.ndarray, shape (nX, nY)
    """
    rng = np.random.default_rng(seed)
    heights = rng.normal(height_mean, height_std, (nX, nY))
    heights = heights - heights.min() + z_safe
    return heights


# ---------------------------------------------------------------------------
# OBJ output
# ---------------------------------------------------------------------------

def generate_terrain_obj(
    nX: int,
    nY: int,
    sL: float,
    height_mean: float,
    height_std: float,
    z_safe: float,
    seed: int | None = None,
    output_path: str | Path = "terrain",
) -> np.ndarray:
    """Generate a terrain OBJ from random tile heights.

    Parameters
    ----------
    nX, nY : int
        Number of tiles in the X and Y directions.
    sL : float
        Side length of each square tile.
    height_mean, height_std : float
        Mean and std-dev for tile height sampling (``N(mean, std)``).
    z_safe : float
        Minimum material thickness — the shortest tile will have this height.
    seed : int, optional
        Random seed for reproducibility.
    output_path : str or Path
        Base path (without extension).  Writes ``{output_path}.obj``.

    Returns
    -------
    heights : np.ndarray, shape (nX, nY)
        The final heightmap (after shifting for z_safe).
    """
    heights = generate_heightmap(nX, nY, height_mean, height_std, z_safe, seed)
    verts, faces = _build_mesh(heights, sL)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_obj(verts, faces, out.with_suffix(".obj"))

    return heights


# ---------------------------------------------------------------------------
# MuJoCo hfield output
# ---------------------------------------------------------------------------

def generate_terrain_hfield(
    nX: int,
    nY: int,
    sL: float,
    height_mean: float,
    height_std: float,
    z_safe: float,
    seed: int | None = None,
    output_path: str | Path = "terrain",
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Generate a MuJoCo-compatible hfield PNG from random tile heights.

    Writes a grayscale PNG where pixel brightness encodes normalised height.
    Returns the heightmap and the ``size`` 4-tuple to use in the MJCF
    ``<hfield>`` element::

        <hfield name="terrain" file="terrain.png"
                size="{x_half} {y_half} {z_top} {z_bottom}" />

    Parameters
    ----------
    nX, nY, sL, height_mean, height_std, z_safe, seed
        Same as ``generate_terrain_obj``.
    output_path : str or Path
        Base path (without extension).  Writes ``{output_path}.png``.

    Returns
    -------
    heights : np.ndarray, shape (nX, nY)
    size : tuple (x_half, y_half, z_top, z_bottom)
        Ready to drop into an MJCF ``<hfield size="...">`` attribute.
    """
    heights = generate_heightmap(nX, nY, height_mean, height_std, z_safe, seed)

    z_top = float(heights.max())
    z_bottom = 0.0
    x_half = nX * sL / 2.0
    y_half = nY * sL / 2.0
    size = (x_half, y_half, z_top, z_bottom)

    # Normalise to [0, 1] then map to uint16 for precision
    data = heights / z_top
    img = (data * np.iinfo(np.uint16).max).astype(np.uint16)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out.with_suffix(".png"), img)

    return heights, size


# ---------------------------------------------------------------------------
# Mesh construction (fully vectorized)
# ---------------------------------------------------------------------------

def _build_mesh(
    heights: np.ndarray, sL: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build a watertight quad mesh for the tiled terrain block.

    Vertex layout
    ~~~~~~~~~~~~~
    Block 0 — top tile corners : 4 per tile, ``(i*nY+j)*4 + {0,1,2,3}``
        v0=(x0,y0,h)  v1=(x1,y0,h)  v2=(x1,y1,h)  v3=(x0,y1,h)
    Block 1 — bottom grid      : ``(nX+1)*(nY+1)`` at z=0

    Returns (vertices, faces) where faces index into the vertex array (0-based).
    """
    nX, nY = heights.shape
    n_tiles = nX * nY

    # --- Top tile vertices (4 per tile) ---
    ii, jj = np.meshgrid(np.arange(nX), np.arange(nY), indexing="ij")
    x0 = (ii * sL).ravel()
    y0 = (jj * sL).ravel()
    x1 = ((ii + 1) * sL).ravel()
    y1 = ((jj + 1) * sL).ravel()
    h = heights.ravel()

    top_v = np.empty((n_tiles * 4, 3))
    top_v[0::4] = np.c_[x0, y0, h]
    top_v[1::4] = np.c_[x1, y0, h]
    top_v[2::4] = np.c_[x1, y1, h]
    top_v[3::4] = np.c_[x0, y1, h]

    # --- Bottom grid vertices ---
    gi, gj = np.meshgrid(np.arange(nX + 1), np.arange(nY + 1), indexing="ij")
    bot_v = np.c_[(gi * sL).ravel(), (gj * sL).ravel(), np.zeros((nX + 1) * (nY + 1))]

    verts = np.vstack([top_v, bot_v])
    B = n_tiles * 4  # index offset for bottom-grid block

    # --- Index helpers (vectorised over i,j arrays) ---
    def tv(i, j, c):
        return (i * nY + j) * 4 + c

    def bv(i, j):
        return B + i * (nY + 1) + j

    # === FACES (all quads, 0-indexed) ===
    face_blocks: list[np.ndarray] = []

    # Top faces  (normal ~ +Z)
    t = np.arange(n_tiles)
    face_blocks.append(np.c_[t * 4, t * 4 + 1, t * 4 + 2, t * 4 + 3])

    # Bottom faces  (normal ~ -Z, reversed winding)
    bi, bj = np.meshgrid(np.arange(nX), np.arange(nY), indexing="ij")
    bi, bj = bi.ravel(), bj.ravel()
    face_blocks.append(np.c_[bv(bi, bj), bv(bi, bj + 1), bv(bi + 1, bj + 1), bv(bi + 1, bj)])

    # X-direction step walls  (between tile (i,j) and tile (i+1,j))
    if nX > 1:
        xi, xj = np.meshgrid(np.arange(nX - 1), np.arange(nY), indexing="ij")
        xi, xj = xi.ravel(), xj.ravel()
        mask = heights[:-1, :].ravel() != heights[1:, :].ravel()
        xi, xj = xi[mask], xj[mask]
        if len(xi):
            face_blocks.append(
                np.c_[tv(xi + 1, xj, 0), tv(xi + 1, xj, 3), tv(xi, xj, 2), tv(xi, xj, 1)]
            )

    # Y-direction step walls  (between tile (i,j) and tile (i,j+1))
    if nY > 1:
        yi, yj = np.meshgrid(np.arange(nX), np.arange(nY - 1), indexing="ij")
        yi, yj = yi.ravel(), yj.ravel()
        mask = heights[:, :-1].ravel() != heights[:, 1:].ravel()
        yi, yj = yi[mask], yj[mask]
        if len(yi):
            face_blocks.append(
                np.c_[tv(yi, yj, 3), tv(yi, yj, 2), tv(yi, yj + 1, 1), tv(yi, yj + 1, 0)]
            )

    # Perimeter walls — connect top tile edges to bottom grid at z=0
    fi = np.arange(nX)
    lj = np.arange(nY)

    # Front (y=0)
    face_blocks.append(np.c_[bv(fi, 0), bv(fi + 1, 0), tv(fi, 0, 1), tv(fi, 0, 0)])
    # Back (y=nY)
    face_blocks.append(np.c_[tv(fi, nY - 1, 3), tv(fi, nY - 1, 2), bv(fi + 1, nY), bv(fi, nY)])
    # Left (x=0)
    face_blocks.append(np.c_[tv(0, lj, 0), tv(0, lj, 3), bv(0, lj + 1), bv(0, lj)])
    # Right (x=nX)
    face_blocks.append(np.c_[bv(nX, lj), bv(nX, lj + 1), tv(nX - 1, lj, 2), tv(nX - 1, lj, 1)])

    faces = np.vstack(face_blocks)
    return verts, faces


# ---------------------------------------------------------------------------
# OBJ writer
# ---------------------------------------------------------------------------

def _write_obj(verts: np.ndarray, faces: np.ndarray, path: Path) -> None:
    with open(path, "w") as f:
        f.write(f"# terrain mesh: {len(verts)} vertices, {len(faces)} faces\n")
        np.savetxt(f, verts, fmt="v %.6f %.6f %.6f")
        np.savetxt(f, faces + 1, fmt="f %d %d %d %d")  # OBJ is 1-indexed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate rough terrain as OBJ mesh and/or MuJoCo hfield PNG."
    )
    parser.add_argument("--nX", type=int, default=10, help="tiles in X (default: 10)")
    parser.add_argument("--nY", type=int, default=10, help="tiles in Y (default: 10)")
    parser.add_argument("--sL", type=float, default=0.005,
                        help="tile side length in meters (default: 0.005)")
    parser.add_argument("--height-mean", type=float, default=0.002,
                        help="mean tile height in meters (default: 0.002)")
    parser.add_argument("--height-std", type=float, default=0.0005,
                        help="std-dev of tile height in meters (default: 0.0005)")
    parser.add_argument("--z-safe", type=float, default=0.001,
                        help="min material thickness in meters (default: 0.001)")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("-o", "--output", type=str, default="terrain",
                        help="output base path without extension (default: terrain)")
    parser.add_argument("--format", choices=["obj", "hfield", "both"], default="both",
                        help="output format (default: both)")

    args = parser.parse_args()
    common = dict(nX=args.nX, nY=args.nY, sL=args.sL, height_mean=args.height_mean,
                  height_std=args.height_std, z_safe=args.z_safe, seed=args.seed,
                  output_path=args.output)

    if args.format in ("obj", "both"):
        heights = generate_terrain_obj(**common)
        print(f"OBJ    — {heights.shape[0]}x{heights.shape[1]} tiles, "
              f"heights {heights.min():.4f}..{heights.max():.4f} m")

    if args.format in ("hfield", "both"):
        heights, size = generate_terrain_hfield(**common)
        print(f"hfield — size=({' '.join(f'{s:.6f}' for s in size)})")


if __name__ == "__main__":
    _cli()
