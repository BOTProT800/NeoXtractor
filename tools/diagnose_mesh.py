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
    --storage {auto,row_vector,column_vector}
    --role {global_bind,inverse_bind,local_bind}
    --conversion {neox_flip_x,identity}
    --no-trs              bake node matrices instead of writing TRS

The overrides exist to test a hypothesis quickly when the defaults turn out
not to fit a variant.
"""

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

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

CONVERSIONS = {"neox_flip_x": NEOX_TO_GLTF, "identity": IDENTITY_CONVERSION}


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
        "--conversion", choices=sorted(CONVERSIONS), default="neox_flip_x"
    )
    parser.add_argument("--no-trs", action="store_true")
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

    destination = Path(args.out) if args.out else source.with_suffix(".glb")
    destination.write_bytes(glb.convert(mesh, **options))
    print(f"\n  wrote {destination} ({destination.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
