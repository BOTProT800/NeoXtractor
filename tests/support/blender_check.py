"""
Verify an exported GLB by importing it into Blender.

Run with the ``bpy`` module installed (Blender as a Python package)::

    uv venv verify-venv
    uv pip install --python verify-venv/Scripts/python.exe bpy
    verify-venv/Scripts/python.exe tests/support/blender_check.py model.glb

Blender's glTF importer is an independent consumer of the file: it resolves the
skin, builds an armature and binds vertex groups without any knowledge of this
project. Agreeing with it is much stronger evidence than agreeing with our own
reader.

Two conventions have to be accounted for:

* Blender is Z-up, glTF is Y-up. The importer rotates the scene, so a glTF
  point ``(x, y, z)`` lands at ``(x, -z, y)``.
* Blender bones have their own local axes (and a roll), unrelated to the glTF
  node axes. Poses are therefore driven through ``pose_bone.matrix``, which is
  expressed in **armature space**, so the delta applied is the same one the
  maths predicts regardless of how Blender oriented the bone.

Driving bone ``j`` with an armature-space delta ``D`` gives
``G_pose[j] = D @ G_rest[j]``. Descendants inherit it, so for every bone in
that subtree ``S = D @ G_rest @ inverse(G_rest) = D``: a vertex influenced only
by the subtree must move by exactly ``D``, and every other vertex must not move
at all. That prediction does not depend on any convention.

The script prints ``RESULT ok`` or ``RESULT failed`` plus one line per check.
"""

import os
import sys

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one named check."""
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}{(' -- ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def gltf_to_blender(point):
    """Map a glTF (Y-up) point onto Blender's (Z-up) axes."""
    return (point[0], -point[2], point[1])


def main(glb_path: str) -> int:
    import bpy
    import numpy as np
    from mathutils import Matrix

    print(f"Blender {bpy.app.version_string}")
    print(f"File    {glb_path}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb_path)

    # Blender's importer leaves a scratch "Icosphere" behind; the mesh that
    # matters is the one carrying an armature modifier.
    skinned = [
        obj
        for obj in bpy.data.objects
        if obj.type == "MESH" and any(m.type == "ARMATURE" for m in obj.modifiers)
    ]
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    check("a skinned mesh was imported", len(skinned) == 1, f"{len(skinned)} found")
    check("an armature was imported", len(armatures) == 1, f"{len(armatures)} found")
    if not skinned or not armatures:
        return 1

    mesh_object = skinned[0]
    armature_object = armatures[0]
    armature = armature_object.data

    modifiers = [m for m in mesh_object.modifiers if m.type == "ARMATURE"]
    check(
        "the armature modifier points at the imported armature",
        modifiers[0].object is armature_object,
    )

    def evaluated_positions():
        """World-space vertex positions after the armature deforms the mesh."""
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = mesh_object.evaluated_get(depsgraph)
        data = evaluated.to_mesh()
        coordinates = np.array(
            [(evaluated.matrix_world @ vertex.co)[:] for vertex in data.vertices]
        )
        evaluated.to_mesh_clear()
        return coordinates

    rest = evaluated_positions()
    print(f"  bones: {len(armature.bones)}  vertices: {len(rest)}")

    # --- the file's own numbers, read back independently ---------------------
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from tests.support.gltf_reader import read_any
    from tests.support.skinning import global_node_matrices

    with open(glb_path, "rb") as handle:
        document = read_any(handle.read())
    gltf = document.json_data
    skin = gltf["skins"][0]
    node_globals = global_node_matrices(gltf)

    primitive = gltf["meshes"][0]["primitives"][0]
    gltf_positions = document.accessor(primitive["attributes"]["POSITION"])
    gltf_joints = document.accessor(primitive["attributes"]["JOINTS_0"]).astype(int)
    gltf_weights = document.accessor(primitive["attributes"]["WEIGHTS_0"]).astype(float)

    diagonal = float(
        np.linalg.norm(gltf_positions.max(axis=0) - gltf_positions.min(axis=0))
    )
    tolerance = max(diagonal * 1e-4, 1e-4)

    check(
        "bone count matches the skin palette",
        len(armature.bones) == len(skin["joints"]),
        f"{len(armature.bones)} vs {len(skin['joints'])}",
    )

    expected_rest = np.array([gltf_to_blender(p) for p in gltf_positions])

    # Blender's importer drops vertices that no triangle references, so the
    # index spaces need not line up. Match on position instead.
    def match_vertices(blender_coords):
        """Map each Blender vertex index onto the exported vertex index."""
        mapping = {}
        worst = 0.0
        for index, coordinate in enumerate(blender_coords):
            distances = np.linalg.norm(expected_rest - coordinate, axis=1)
            nearest = int(distances.argmin())
            mapping[index] = nearest
            worst = max(worst, float(distances[nearest]))
        return mapping, worst

    vertex_map, worst_match = match_vertices(rest)
    dropped = len(expected_rest) - len(rest)
    check(
        "rest geometry matches the exported positions",
        worst_match <= tolerance,
        f"max error {worst_match:.3e}, tolerance {tolerance:.3e}"
        + (f"; Blender dropped {dropped} unreferenced vertices" if dropped else ""),
    )

    # --- pivots: where Blender thinks each joint sits ------------------------
    pivot_errors = {}
    for slot, node_index in enumerate(skin["joints"]):
        name = gltf["nodes"][node_index]["name"]
        if name not in armature.bones:
            pivot_errors[name] = "missing"
            continue
        expected = np.array(gltf_to_blender(node_globals[node_index][:3, 3]))
        actual = np.array(
            (armature_object.matrix_world @ armature.bones[name].matrix_local).translation[:]
        )
        error = float(np.linalg.norm(actual - expected))
        if error > tolerance:
            pivot_errors[name] = f"{error:.4f} at {actual.tolist()} expected {expected.tolist()}"
    check(
        "every bone head sits at its exported joint origin",
        not pivot_errors,
        "; ".join(f"{k}: {v}" for k, v in list(pivot_errors.items())[:4]),
    )

    origins = np.array(
        [
            (armature_object.matrix_world @ bone.matrix_local).translation[:]
            for bone in armature.bones
        ]
    )
    check(
        "bones are not all collapsed onto the origin",
        float(np.abs(origins).max()) > tolerance,
        f"largest bone offset {float(np.abs(origins).max()):.4f}",
    )

    # --- parents -------------------------------------------------------------
    parent_errors = []
    node_name = {index: node.get("name") for index, node in enumerate(gltf["nodes"])}
    child_of = {}
    for index, node in enumerate(gltf["nodes"]):
        for child in node.get("children", []):
            child_of[child] = index
    for node_index in skin["joints"]:
        name = node_name[node_index]
        if name not in armature.bones:
            continue
        expected_parent = node_name.get(child_of.get(node_index))
        actual_parent = armature.bones[name].parent
        actual_name = actual_parent.name if actual_parent else None
        if expected_parent in {node_name[n] for n in skin["joints"]}:
            if actual_name != expected_parent:
                parent_errors.append(f"{name}: {actual_name} != {expected_parent}")
        elif actual_name is not None:
            parent_errors.append(f"{name}: got parent {actual_name}, expected none")
    check(
        "the bone hierarchy matches the exported parents",
        not parent_errors,
        "; ".join(parent_errors[:4]),
    )

    # --- weights -------------------------------------------------------------
    group_name = {group.index: group.name for group in mesh_object.vertex_groups}
    weight_errors = []
    for vertex in mesh_object.data.vertices:
        source = vertex_map[vertex.index]
        blender_weights = {
            group_name[element.group]: element.weight for element in vertex.groups
        }
        exported = {}
        for slot, weight in zip(gltf_joints[source], gltf_weights[source]):
            if weight > 0.0:
                exported[node_name[skin["joints"][slot]]] = (
                    exported.get(node_name[skin["joints"][slot]], 0.0) + float(weight)
                )
        if set(blender_weights) != set(exported):
            weight_errors.append(
                f"v{vertex.index}: {sorted(blender_weights)} != {sorted(exported)}"
            )
            continue
        for bone_name, weight in exported.items():
            if abs(blender_weights[bone_name] - weight) > 1e-4:
                weight_errors.append(
                    f"v{vertex.index}/{bone_name}: {blender_weights[bone_name]:.4f} != {weight:.4f}"
                )
    check(
        "vertex groups reproduce the exported influences",
        not weight_errors,
        "; ".join(weight_errors[:4]),
    )

    # --- deformation ---------------------------------------------------------
    # Descendants of each joint, by bone name.
    subtree = {name: {name} for name in [node_name[n] for n in skin["joints"]]}
    for node_index in skin["joints"]:
        cursor = child_of.get(node_index)
        while cursor is not None:
            if node_name[cursor] in subtree:
                subtree[node_name[cursor]].add(node_name[node_index])
            cursor = child_of.get(cursor)

    bone_names = [node_name[n] for n in skin["joints"] if node_name[n] in armature.bones]
    # Prefer a bone that has children and is not the only root.
    targets = sorted(bone_names, key=lambda n: -len(subtree[n]))
    targets = [t for t in targets if len(subtree[t]) < len(bone_names)][:3] or targets[:1]

    for target in targets:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=glb_path)
        mesh_object = [
            o
            for o in bpy.data.objects
            if o.type == "MESH" and any(m.type == "ARMATURE" for m in o.modifiers)
        ][0]
        armature_object = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]
        group_name = {group.index: group.name for group in mesh_object.vertex_groups}
        rest_now = evaluated_positions()

        # A 90 degree turn about armature-space Z, through the bone's head.
        pose_bone = armature_object.pose.bones[target]
        head = armature_object.data.bones[target].matrix_local.translation.copy()
        delta = (
            Matrix.Translation(head)
            @ Matrix.Rotation(np.radians(90.0), 4, "Z")
            @ Matrix.Translation(-head)
        )
        pose_bone.matrix = delta @ armature_object.data.bones[target].matrix_local
        bpy.context.view_layer.update()

        applied = np.array(pose_bone.matrix)
        wanted = np.array(delta @ armature_object.data.bones[target].matrix_local)
        check(
            f"pose delta applied to {target} round-trips",
            np.allclose(applied, wanted, atol=1e-4),
            f"max error {np.abs(applied - wanted).max():.3e}",
        )

        posed = evaluated_positions()

        # The armature-space delta seen from world space.
        world_delta = np.array(
            armature_object.matrix_world
            @ delta
            @ armature_object.matrix_world.inverted()
        )

        moved_wrong = []
        for vertex in mesh_object.data.vertices:
            index = vertex.index
            influencing = {
                group_name[e.group] for e in vertex.groups if e.weight > 0.0
            }
            inside = influencing & subtree[target]
            if influencing and inside == influencing:
                # Wholly inside the subtree: the delta applies exactly.
                expected = world_delta @ np.append(rest_now[index], 1.0)
                if not np.allclose(posed[index], expected[:3], atol=tolerance):
                    moved_wrong.append(
                        f"v{index} in subtree: {posed[index].tolist()} != {expected[:3].tolist()}"
                    )
            elif not inside:
                if not np.allclose(posed[index], rest_now[index], atol=tolerance):
                    moved_wrong.append(
                        f"v{index} outside subtree moved by "
                        f"{np.linalg.norm(posed[index] - rest_now[index]):.4f}"
                    )
        check(
            f"rotating {target} moves exactly its subtree",
            not moved_wrong,
            "; ".join(moved_wrong[:4]),
        )

        any_moved = np.abs(posed - rest_now).max() > tolerance
        check(f"rotating {target} deformed something", bool(any_moved))

    print(f"RESULT {'ok' if not FAILURES else 'failed: ' + ', '.join(FAILURES)}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: blender_check.py <file.glb>")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
