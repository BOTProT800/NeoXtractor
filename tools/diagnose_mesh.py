"""
Export a .mesh and report everything the exporter decided about it.

Run from the repository root::

    uv run python tools/diagnose_mesh.py ruta/al/modelo.mesh

It writes ``<modelo>.glb`` next to the input and prints what the parser read,
what convention the skeleton module detected and why, and any diagnostic the
export raised. That report is what makes a "it looks wrong" observation
actionable: it says which variant the file is, how its matrices were
interpreted, and whether anything was dropped.

Options::

    --out PATH            write the .glb somewhere else
    --anim FILE.gis       attach animation clips, matched by bone name
    --clips a,b,c         only these clips
    --storage {auto,row_vector,column_vector}
    --role {global_bind,inverse_bind,local_bind}
    --conversion {identity,mirror_x}
    --no-trs              bake node matrices instead of writing TRS
    --validate            run the Khronos glTF-Validator on the result
                          (needs NEOX_GLTF_VALIDATOR, see tests/support/khronos.py)
    --render [CLIP]       draw the result to <modelo>.png, with frames of CLIP
                          if given (needs NEOX_BLENDER_PYTHON, see
                          tools/render_glb.py)

The overrides exist to test a hypothesis quickly when the defaults turn out
not to fit a variant. The default, ``mirror_x``, is right for NeoX, which is
left-handed; ``--conversion identity`` writes the stored coordinates
unchanged, which is the mirror image, for comparing the two side by side.
See ``NEOX_TO_GLTF`` for the evidence; the handedness report below repeats
the check on any rig with left/right bone names.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from core.anim_loader import is_rgis, read_rgis  # noqa: E402
from core.mesh_converter.animation import clips_from_rgis  # noqa: E402
from core.mesh_converter.formats import glb  # noqa: E402
from core.mesh_converter.gltf_scene import build_scene  # noqa: E402
from core.mesh_converter.skeleton import (  # noqa: E402
    IDENTITY_CONVERSION,
    NEOX_TO_GLTF,
    MatrixRole,
    MatrixStorage,
    detect_matrix_storage,
)
from core.mesh_loader import MeshLoader  # noqa: E402

#: Keyed by the conversion's own name, so the label cannot drift from what it does.
CONVERSIONS = {
    conversion.name: conversion for conversion in (NEOX_TO_GLTF, IDENTITY_CONVERSION)
}


def heading(text):
    print()
    print(text)
    print("-" * len(text))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", help="path to the .mesh file")
    parser.add_argument("--out", help="where to write the .glb")
    parser.add_argument(
        "--storage",
        choices=[item.value for item in MatrixStorage],
        default=MatrixStorage.AUTO.value,
    )
    parser.add_argument(
        "--role",
        choices=[item.value for item in MatrixRole],
        default=MatrixRole.GLOBAL_BIND.value,
    )
    parser.add_argument(
        "--conversion", choices=sorted(CONVERSIONS), default=NEOX_TO_GLTF.name
    )
    parser.add_argument("--no-trs", action="store_true")
    parser.add_argument(
        "--anim",
        help="a .gis animation container to attach (bones are matched by name)",
    )
    parser.add_argument(
        "--clips",
        help="comma separated clip names to include; default is all of them",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="run the Khronos glTF-Validator on the written file",
    )
    parser.add_argument(
        "--render",
        nargs="?",
        const="",
        metavar="CLIP",
        help="render the written file to a picture, with frames of CLIP if given",
    )
    args = parser.parse_args()

    source = Path(args.mesh)
    if not source.is_file():
        print(f"not a file: {source}")
        return 2

    payload = source.read_bytes()
    heading("File")
    print(f"  path    {source}")
    print(f"  size    {len(payload)} bytes")
    print(f"  sha256  {hashlib.sha256(payload).hexdigest()}")

    mesh = MeshLoader().load_from_bytes(payload)
    if mesh is None:
        print("\n  The parser could not read this file. The log above says why.")
        return 1

    heading("What the parser read")
    print(f"  version            {mesh.version}")
    print(f"  variant (type)     {mesh.type}")
    print(f"  bone kind          {mesh.bones.has_bones}")
    print(f"  vertices           {mesh.mesh.vertexes}")
    print(f"  faces              {mesh.mesh.faces}")
    print(f"  uv entries         {len(mesh.mesh.uv)}")
    print(f"  bones              {len(mesh.bones.names)} (count field {mesh.bones.count})")
    print(f"  joint index width  {mesh.bones.joint_index_bits} bits "
          f"(empty slot = {mesh.bones.joint_index_sentinel})")

    geometry_problems = mesh.validate_geometry()
    rig_problems = mesh.validate_rig()
    print(f"  geometry problems  {len(geometry_problems)}")
    for problem in geometry_problems[:6]:
        print(f"    - {problem}")
    print(f"  rig problems       {len(rig_problems)}")
    for problem in rig_problems[:6]:
        print(f"    - {problem}")

    if mesh.bones.names:
        heading("Skeleton")
        roots = [i for i, p in enumerate(mesh.bones.parents) if p == -1]
        print(f"  roots              {len(roots)} -> "
              f"{[mesh.bones.names[i] for i in roots][:6]}")
        out_of_order = sum(
            1 for i, p in enumerate(mesh.bones.parents) if p != -1 and p > i
        )
        print(f"  parents after child {out_of_order}")

        storage, reason = detect_matrix_storage(mesh.bones.matrix)
        print(f"  detected storage   {storage.value}")
        print(f"    because          {reason}")

        weights = np.asarray(mesh.bones.weights, dtype=np.float64)
        if len(weights):
            sums = weights.sum(axis=1)
            print(f"  weight sums        min {sums.min():.6f} max {sums.max():.6f}")
            print(f"  vertices with no weight  {int(np.count_nonzero(sums <= 0.0))}")
        joints = np.asarray(mesh.bones.joints, dtype=np.int64)
        if len(joints):
            print(f"  joint index range  {int(joints.min())}..{int(joints.max())}")

        heading("First bones (origin under the detected storage)")
        for index in range(min(8, len(mesh.bones.names))):
            matrix = np.asarray(mesh.bones.matrix[index], dtype=np.float64)
            origin = matrix[3, :3] if storage is MatrixStorage.ROW_VECTOR else matrix[:3, 3]
            parent = mesh.bones.parents[index]
            parent_name = mesh.bones.names[parent] if parent != -1 else "-"
            print(f"  {index:>4} {mesh.bones.names[index][:28]:<28} "
                  f"parent {parent_name[:20]:<20} {np.round(origin, 4).tolist()}")

    heading("Export")
    options = {
        "conversion": CONVERSIONS[args.conversion],
        "matrix_storage": MatrixStorage(args.storage),
        "matrix_role": MatrixRole(args.role),
        "use_trs_nodes": not args.no_trs,
    }
    try:
        scene = build_scene(mesh, **options)
    except Exception as error:
        print(f"  FAILED: {type(error).__name__}: {error}")
        return 1

    for note in scene.diagnostics:
        print(f"  - {note}")
    print(f"  skin attached      {scene.is_skinned}")
    print(f"  joints in palette  {len(scene.slot_to_node)}")

    clips = []
    if args.anim:
        heading("Animation")
        animation_path = Path(args.anim)
        payload = animation_path.read_bytes()
        if not is_rgis(payload):
            print(f"  {animation_path} is not an RGIS container "
                  f"(magic {payload[:4]!r}); skipped")
        elif scene.skeleton is None:
            print("  the mesh has no usable skeleton, so clips cannot be attached")
        else:
            rgis = read_rgis(payload)
            print(f"  file      {animation_path.name}  version {rgis.version}")
            print(f"  reference {len(rgis.reference_names)} bones")
            print(f"  clips     {len(rgis.clips)} read, {len(rgis.skipped)} skipped")
            for name, reason in rgis.skipped:
                print(f"    SKIPPED {name}: {reason}")

            wanted = args.clips.split(",") if args.clips else None
            clips, notes = clips_from_rgis(rgis, scene.skeleton, only=wanted)
            for note in notes:
                print(f"    - {note}")
            for clip in clips:
                print(f"    {clip.name:<20} tracks={len(clip.tracks):<3} "
                      f"{clip.duration:.3f}s")
            options["animations"] = clips

    destination = Path(args.out) if args.out else source.with_suffix(".glb")
    destination.write_bytes(glb.convert(mesh, **options))
    print(f"\n  wrote {destination} ({destination.stat().st_size} bytes)")

    status = 0
    if args.validate:
        status = max(status, validate(destination))
    if args.render is not None:
        status = max(status, render(destination, args.render))
    return status


def validate(destination: Path) -> int:
    """Run the official validator; 1 if it reports errors."""
    from tests.support import khronos

    heading("Khronos glTF-Validator")
    if not khronos.available():
        print("  not available: install the gltf-validator npm package, point")
        print("  NEOX_GLTF_VALIDATOR at it and put node on PATH")
        return 1
    (report,) = khronos.validate(destination)
    print(textwrap.indent(khronos.summary(report), "  "))
    return 1 if report["numErrors"] else 0


def render(destination: Path, clip: str) -> int:
    """Draw the export with Blender, so it can be compared with the game."""
    heading("Picture")
    interpreter = os.environ.get("NEOX_BLENDER_PYTHON")
    if not interpreter:
        print("  not available: set NEOX_BLENDER_PYTHON to an interpreter with bpy")
        return 1
    command = [interpreter, str(REPO_ROOT / "tools" / "render_glb.py"), str(destination)]
    if clip:
        command += ["--clip", clip]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    keep = ("clip ", "model ", "bounds ", "wrote ", "no clip", "no mesh", "the file")
    for line in process.stdout.splitlines():
        if line.startswith(keep):
            print(f"  {line}")
    if process.returncode != 0:
        print(textwrap.indent(process.stderr[-2000:], "  "))
        return 1
    print("  Nothing here can tell a mirror image from the real thing. Compare")
    print("  something asymmetric with the game before calling it good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
