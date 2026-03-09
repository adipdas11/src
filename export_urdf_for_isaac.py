#!/usr/bin/env python3
"""
export_urdf_for_isaac.py

Creates a self-contained folder for Isaac Sim import:
  <output_dir>/
    dual_arm_world.urdf          ← rewritten URDF (all paths relative)
    meshes/
      rg6_description/visual/base_link.stl
      xarm5_description/visual/link1.stl
      ...

Copies every referenced mesh into the output folder (renaming hyphens to
underscores where needed). The original workspace is never modified.

Usage:
    python3 export_urdf_for_isaac.py

    Then in Isaac Sim, import:
        <output_dir>/dual_arm_world.urdf
"""
import re
import shutil
from pathlib import Path

# ─── CONFIGURATION ───────────────────────────────────────────────────────────

WS = Path.home() / "workspaces" / "disassembly_ws"
WS_SRC = WS / "src"

# Input URDF (the xacro-generated flat file)
INPUT_URDF = (
    WS_SRC
    / "dual_arm_moveit_config"
    / "config"
    / "dual_arm_world.urdf"
)

# Output directory — inside src/ alongside your other packages
OUTPUT_DIR = WS_SRC / "isaac_export"

# package:// name → source directory (the folder containing meshes/)
PACKAGE_MAP: dict[str, Path] = {
    "xarm5_description":        WS_SRC / "robots" / "xarm5_description",
    "uf850_description":        WS_SRC / "robots" / "uf850_description",
    "uf850_mount_description":  WS_SRC / "robots" / "uf850_mount_description",
    "uf_slider_description":    WS_SRC / "robots" / "uf_slider_description",
    "rg6_description":          WS_SRC / "robots" / "rg6_description",
    "screwdriver_description":  WS_SRC / "robots" / "screwdriver_description",
    "scene_description":        WS_SRC / "scene_description",
}

# file:// absolute paths → pseudo-package name for the export folder
FILE_PACKAGE_ALIAS = "robotiq_ft_sensor_description"
FILE_PACKAGE_ROOT = (
    WS
    / "install"
    / "robotiq_ft_sensor_description"
    / "share"
    / "robotiq_ft_sensor_description"
)


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def sanitise(name: str) -> str:
    """Replace hyphens with underscores (USD prim path compliance)."""
    return name.replace("-", "_")


def copy_mesh(src: Path, dst: Path) -> None:
    """Copy a mesh file, creating parent dirs as needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        print(f"  WARNING: source mesh not found — {src}")
        return
    shutil.copy2(src, dst)


# ─── MAIN ───────────────────────────────────────────────────────────────────

def main() -> None:
    # Resolve input — allow override via command line later if needed
    input_urdf = INPUT_URDF
    if not input_urdf.exists():
        # Fallback: maybe the user has it elsewhere
        alt = WS_SRC / "scene_description" / "urdf" / "dual_arm_world.urdf"
        if alt.exists():
            input_urdf = alt
        else:
            raise FileNotFoundError(
                f"Cannot find URDF at {INPUT_URDF} or {alt}.\n"
                "Edit INPUT_URDF in this script to point at your file."
            )

    out_dir = OUTPUT_DIR
    meshes_dir = out_dir / "meshes"

    # Clean previous export
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    print(f"Input:  {input_urdf}")
    print(f"Output: {out_dir}\n")

    urdf = input_urdf.read_text()

    # ── 1. Handle package:// URIs ────────────────────────────────────────────
    pkg_re = re.compile(r'filename="package://([^/]+)/(.*?)"')

    def replace_package(m: re.Match) -> str:
        pkg = m.group(1)
        rel = m.group(2)  # e.g. meshes/visual/link1.stl

        if pkg not in PACKAGE_MAP:
            print(f"  WARNING: unknown package '{pkg}' — left as-is")
            return m.group(0)

        src_file = PACKAGE_MAP[pkg] / rel
        # Sanitise the filename component only
        parts = Path(rel).parts
        safe_parts = [sanitise(p) for p in parts]
        safe_rel = Path(*safe_parts)

        dst_file = meshes_dir / pkg / safe_rel
        copy_mesh(src_file, dst_file)

        # Relative path from the URDF file to the mesh
        rel_from_urdf = Path("meshes") / pkg / safe_rel
        return f'filename="{rel_from_urdf}"'

    urdf = pkg_re.sub(replace_package, urdf)

    # ── 2. Handle file:// absolute URIs (Robotiq FT sensor) ─────────────────
    file_re = re.compile(r'filename="file://(.*?)"')

    def replace_file(m: re.Match) -> str:
        abs_path = Path(m.group(1))

        # Determine relative path under the package root
        try:
            rel = abs_path.relative_to(FILE_PACKAGE_ROOT)
        except ValueError:
            print(f"  WARNING: file:// path not under known root — {abs_path}")
            return m.group(0)

        # Sanitise every component
        safe_parts = [sanitise(p) for p in rel.parts]
        safe_rel = Path(*safe_parts)

        src_file = abs_path
        dst_file = meshes_dir / FILE_PACKAGE_ALIAS / safe_rel
        copy_mesh(src_file, dst_file)

        rel_from_urdf = Path("meshes") / FILE_PACKAGE_ALIAS / safe_rel
        return f'filename="{rel_from_urdf}"'

    urdf = file_re.sub(replace_file, urdf)

    # ── 3. Sanitise material names (Robotiq-Black → Robotiq_Black) ──────────
    urdf = urdf.replace('"Robotiq-Black"', '"Robotiq_Black"')
    urdf = urdf.replace('"Robotiq-Grey"', '"Robotiq_Grey"')

    # ── 4. Write the URDF ────────────────────────────────────────────────────
    out_urdf = out_dir / "dual_arm_world.urdf"
    out_urdf.write_text(urdf)

    # ── 5. Summary ───────────────────────────────────────────────────────────
    mesh_count = sum(1 for _ in meshes_dir.rglob("*.stl")) + \
                 sum(1 for _ in meshes_dir.rglob("*.STL"))
    print(f"\nDone! Copied {mesh_count} meshes.")
    print(f"URDF:   {out_urdf}")
    print(f"\nIn Isaac Sim → File → Import → select:\n  {out_urdf}")


if __name__ == "__main__":
    main()
