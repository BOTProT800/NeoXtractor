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

    def unpose():
        """
        Drop any imported action so the rig sits at its rest pose.

        With an action linked, Blender evaluates the armature at the current
        frame, so the "rest" geometry would already be animated and every
        pose comparison below would be measured against a moving target.
        Clearing the action is not enough on its own: the pose bones keep
        whatever values the last evaluation left in them, so their basis has
        to be reset too.
        """
        for obj in bpy.data.objects:
            if obj.animation_data:
                obj.animation_data_clear()
            if obj.type == "ARMATURE":
                for pose_bone in obj.pose.bones:
                    pose_bone.matrix_basis = Matrix()
        bpy.context.view_layer.update()

    unpose()

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

    # When Blender kept every vertex it also kept their order, and an
    # index-for-index comparison is exact. Only fall back to matching on
    # position when it dropped unreferenced vertices, and say which happened.
    dropped = len(expected_rest) - len(rest)
    if dropped == 0 and np.allclose(rest, expected_rest, atol=tolerance):
        vertex_map = {index: index for index in range(len(rest))}
        worst_match = float(np.abs(rest - expected_rest).max())
        how = "index for index"
    else:
        vertex_map, worst_match = match_vertices(rest)
        how = "matched on position"
    check(
        "rest geometry matches the exported positions",
        worst_match <= tolerance,
        f"max error {worst_match:.3e}, tolerance {tolerance:.3e}, {how}"
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

    def exported_influences(index):
        """Bone name -> weight for one exported vertex."""
        out = {}
        for slot, weight in zip(gltf_joints[index], gltf_weights[index]):
            if weight > 0.0:
                bone = node_name[skin["joints"][slot]]
                out[bone] = out.get(bone, 0.0) + float(weight)
        return out

    # Meshes duplicate vertices at UV and normal seams, so several exported
    # vertices can share one position and they need not share weights. Matching
    # a Blender vertex to a single nearest source would then compare against an
    # arbitrary one of them. Group by position and accept a match against any
    # member; a genuine weighting error still fails, because then no member
    # matches.
    weight_errors = []
    for vertex in mesh_object.data.vertices:
        blender_weights = {
            group_name[element.group]: element.weight for element in vertex.groups
        }
        # Every source vertex sitting at this one's position, found by distance
        # rather than by a rounded key, so a value straddling a rounding
        # boundary cannot silently pick the wrong neighbour.
        distances = np.linalg.norm(expected_rest - rest[vertex.index], axis=1)
        candidates = np.nonzero(distances <= tolerance)[0].tolist()
        if not candidates:
            candidates = [vertex_map[vertex.index]]
        matched = False
        for candidate in candidates:
            exported = exported_influences(candidate)
            if set(blender_weights) != set(exported):
                continue
            if all(
                abs(blender_weights[bone] - weight) <= 1e-4
                for bone, weight in exported.items()
            ):
                matched = True
                break
        if not matched:
            shown = exported_influences(candidates[0])
            weight_errors.append(
                f"v{vertex.index}: blender {sorted(blender_weights)} matches none of "
                f"{len(candidates)} source vertex(es) at that position, e.g. {sorted(shown)}"
            )
    check(
        "vertex groups reproduce the exported influences",
        not weight_errors,
        "; ".join(weight_errors[:3]),
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
        for obj in bpy.data.objects:
            if obj.animation_data:
                obj.animation_data_clear()
            if obj.type == "ARMATURE":
                for pose_bone in obj.pose.bones:
                    pose_bone.matrix_basis = Matrix()
        bpy.context.view_layer.update()
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

    # --- animations ----------------------------------------------------------
    if gltf.get("animations"):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=glb_path)
        mesh_object = [
            o
            for o in bpy.data.objects
            if o.type == "MESH" and any(m.type == "ARMATURE" for m in o.modifiers)
        ][0]

        expected_names = [clip.get("name") for clip in gltf["animations"]]
        actions = list(bpy.data.actions)
        check(
            "every animation was imported as an action",
            len(actions) == len(expected_names),
            f"{[a.name for a in actions]} vs {expected_names}",
        )

        scene = bpy.context.scene

        # Sample whichever action Blender actually has active, over its own
        # frame range. Deriving the range from the first glTF animation instead
        # aliases badly when a different action is the live one.
        armature_object = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]
        animation_data = armature_object.animation_data
        active = animation_data.action if animation_data else None
        check("an action is live on the armature", active is not None)
        if active is None:
            print(f"RESULT {'ok' if not FAILURES else 'failed'}")
            return 0 if not FAILURES else 1

        first_frame, last_frame = (float(v) for v in active.frame_range)
        span = last_frame - first_frame
        print(f"  active action {active.name!r}, frames {first_frame}..{last_frame}")

        def at(fraction):
            scene.frame_set(int(round(first_frame + span * fraction)))
            bpy.context.view_layer.update()
            return evaluated_positions()

        duration = span
        check("the active action spans more than one frame", span > 0.0)

        if span > 0.0:
            start = at(0.0)
            travel = max(
                float(np.abs(at(fraction) - start).max())
                for fraction in (0.25, 0.5, 0.75, 1.0)
            )
            check(
                "the animation actually deforms the mesh",
                travel > tolerance * 10.0,
                f"largest displacement {travel:.4f}",
            )

            # Compare Blender's evaluation against an independent CPU
            # evaluation of the same file at the same instants. Heuristics
            # about the shape of the motion do not hold -- a real idle
            # oscillates, so denser sampling can legitimately show a larger
            # step than coarser sampling -- but two unrelated implementations
            # agreeing is a real invariant.
            from tests.support.skinning import animation_overrides, skin_vertices

            # Find which glTF animation Blender made live.
            live = next(
                (
                    position
                    for position, clip in enumerate(gltf["animations"])
                    if clip.get("name") == active.name
                ),
                0,
            )
            fps = scene.render.fps
            worst = 0.0
            for step in range(0, 9):
                frame = int(round(first_frame + span * step / 8))
                scene.frame_set(frame)
                bpy.context.view_layer.update()
                blender_positions = evaluated_positions()

                # The importer lays keys out at time * fps, so a frame maps
                # straight back onto a time in seconds.
                own = skin_vertices(
                    document, animation_overrides(document, live, frame / fps)
                )
                own_in_blender_axes = np.array([gltf_to_blender(p) for p in own])
                mapped = np.array(
                    [own_in_blender_axes[vertex_map[i]] for i in range(len(blender_positions))]
                )
                worst = max(worst, float(np.abs(blender_positions - mapped).max()))

            check(
                "Blender's posed mesh matches an independent CPU evaluation",
                worst <= max(tolerance * 20.0, 1e-3),
                f"largest disagreement {worst:.6f} over 9 instants of "
                f"{active.name!r}",
            )

            # Every animated bone, across every clip, must be one the skin
            # actually uses.
            animated = {
                gltf["nodes"][channel["target"]["node"]].get("name")
                for clip in gltf["animations"]
                for channel in clip["channels"]
                if channel["target"].get("node") is not None
            }
            joint_names = {
                gltf["nodes"][node].get("name") for node in skin["joints"]
            }
            check(
                "animated nodes are joints of the skin",
                animated <= joint_names,
                f"not joints: {sorted(animated - joint_names)}",
            )

    print(f"RESULT {'ok' if not FAILURES else 'failed: ' + ', '.join(FAILURES)}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: blender_check.py <file.glb>")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
