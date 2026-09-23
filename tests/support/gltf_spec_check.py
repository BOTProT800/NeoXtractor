"""
Structural checks against the glTF 2.0 specification.

This is **not** the Khronos glTF-Validator. It re-implements the subset of that
tool's rules this pipeline can actually break -- accessor bounds and alignment,
skin and joint indices, weight normalisation, unit normals, node cycles --
directly from the specification text, so the test suite can run offline.
Running the official validator on real exports is still worth doing and stays
on the pending list.

Rule references point at the glTF 2.0 specification:
https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
"""

from __future__ import annotations

import numpy as np

from .gltf_reader import ParsedGLTF

_COMPONENT_SIZE = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}

#: Tolerance for the "weights must sum to 1" rule. The official validator uses
#: a comparable bound on the stored float32 values.
WEIGHT_SUM_TOLERANCE = 2e-7

#: Tolerance for the "NORMAL must be unit length" rule.
UNIT_LENGTH_TOLERANCE = 1e-5


def check_gltf(document: ParsedGLTF) -> list[str]:
    """
    Run the structural checks.

    Parameters:
    - document: a parsed glTF or GLB file.

    Returns:
    - A list of specification violations. Empty means the file passes every
      rule implemented here.
    """
    problems: list[str] = []
    gltf = document.json_data

    problems += _check_asset(gltf)
    problems += _check_buffer_views(gltf, document)
    problems += _check_accessors(gltf, document)
    problems += _check_nodes(gltf)
    problems += _check_meshes(gltf, document)
    problems += _check_skins(gltf, document)
    problems += _check_scenes(gltf)
    return problems


def _check_asset(gltf: dict) -> list[str]:
    asset = gltf.get("asset")
    if not isinstance(asset, dict):
        return ["asset is required"]
    if asset.get("version") != "2.0":
        return [f"asset.version is {asset.get('version')!r}, expected '2.0'"]
    return []


def _check_buffer_views(gltf: dict, document: ParsedGLTF) -> list[str]:
    problems: list[str] = []
    buffers = gltf.get("buffers", [])
    for index, view in enumerate(gltf.get("bufferViews", [])):
        buffer_index = view.get("buffer", 0)
        if buffer_index >= len(buffers):
            problems.append(f"bufferView {index} refers to missing buffer {buffer_index}")
            continue
        offset = view.get("byteOffset", 0)
        length = view["byteLength"]
        declared = buffers[buffer_index]["byteLength"]
        if offset + length > declared:
            problems.append(
                f"bufferView {index} spans [{offset}, {offset + length}) "
                f"past the {declared} byte buffer"
            )
        if buffer_index < len(document.buffers) and offset + length > len(
            document.buffers[buffer_index]
        ):
            problems.append(
                f"bufferView {index} spans past the {len(document.buffers[buffer_index])} "
                "bytes actually present"
            )
    return problems


def _check_accessors(gltf: dict, document: ParsedGLTF) -> list[str]:
    problems: list[str] = []
    views = gltf.get("bufferViews", [])
    for index, accessor in enumerate(gltf.get("accessors", [])):
        component_type = accessor.get("componentType")
        if component_type not in _COMPONENT_SIZE:
            problems.append(f"accessor {index} has unknown componentType {component_type}")
            continue
        type_str = accessor.get("type")
        if type_str not in _TYPE_COMPONENTS:
            problems.append(f"accessor {index} has unknown type {type_str!r}")
            continue

        component_size = _COMPONENT_SIZE[component_type]
        element_size = component_size * _TYPE_COMPONENTS[type_str]
        count = accessor.get("count", 0)
        if count < 1:
            problems.append(f"accessor {index} has count {count}, minimum is 1")

        accessor_offset = accessor.get("byteOffset", 0)
        # Spec: accessor.byteOffset MUST be a multiple of the component size.
        if accessor_offset % component_size:
            problems.append(
                f"accessor {index} byteOffset {accessor_offset} is not a multiple "
                f"of its {component_size} byte component"
            )

        if "bufferView" not in accessor:
            continue
        view_index = accessor["bufferView"]
        if view_index >= len(views):
            problems.append(f"accessor {index} refers to missing bufferView {view_index}")
            continue
        view = views[view_index]
        view_offset = view.get("byteOffset", 0)
        # Spec: the effective offset MUST be a multiple of the component size.
        if (view_offset + accessor_offset) % component_size:
            problems.append(
                f"accessor {index} starts at byte {view_offset + accessor_offset}, "
                f"misaligned for its {component_size} byte component"
            )
        stride = view.get("byteStride") or element_size
        needed = accessor_offset + stride * (count - 1) + element_size
        if needed > view["byteLength"]:
            problems.append(
                f"accessor {index} needs {needed} bytes of bufferView {view_index} "
                f"which is only {view['byteLength']} long"
            )

        if accessor.get("type") == "VEC3" and "min" in accessor:
            decoded = document.accessor(index)
            if len(decoded):
                actual_min = decoded.min(axis=0)
                actual_max = decoded.max(axis=0)
                if not np.allclose(actual_min, accessor["min"], atol=1e-5):
                    problems.append(
                        f"accessor {index} declares min {accessor['min']} "
                        f"but the data minimum is {actual_min.tolist()}"
                    )
                if "max" in accessor and not np.allclose(
                    actual_max, accessor["max"], atol=1e-5
                ):
                    problems.append(
                        f"accessor {index} declares max {accessor['max']} "
                        f"but the data maximum is {actual_max.tolist()}"
                    )
    return problems


def _check_nodes(gltf: dict) -> list[str]:
    problems: list[str] = []
    nodes = gltf.get("nodes", [])
    parent_of: dict[int, int] = {}

    for index, node in enumerate(nodes):
        if "matrix" in node and any(key in node for key in ("translation", "rotation", "scale")):
            problems.append(f"node {index} mixes matrix with TRS properties")
        if "matrix" in node and len(node["matrix"]) != 16:
            problems.append(f"node {index} matrix has {len(node['matrix'])} values, expected 16")
        if "rotation" in node:
            rotation = np.asarray(node["rotation"], dtype=np.float64)
            if rotation.shape != (4,):
                problems.append(f"node {index} rotation must hold 4 values")
            elif abs(float(np.linalg.norm(rotation)) - 1.0) > 1e-5:
                problems.append(
                    f"node {index} rotation is not a unit quaternion "
                    f"(length {float(np.linalg.norm(rotation)):.6f})"
                )
        for child in node.get("children", []):
            if child >= len(nodes):
                problems.append(f"node {index} refers to missing child {child}")
                continue
            if child == index:
                problems.append(f"node {index} is its own child")
            if child in parent_of:
                problems.append(
                    f"node {child} is a child of both {parent_of[child]} and {index}"
                )
            parent_of[child] = index

    # Spec: the node hierarchy MUST be a set of disjoint strict trees.
    for start in range(len(nodes)):
        seen = {start}
        cursor = parent_of.get(start)
        while cursor is not None:
            if cursor in seen:
                problems.append(f"node {start} sits on a parent cycle through {cursor}")
                break
            seen.add(cursor)
            cursor = parent_of.get(cursor)
    return problems


def _check_meshes(gltf: dict, document: ParsedGLTF) -> list[str]:
    problems: list[str] = []
    accessors = gltf.get("accessors", [])

    for mesh_index, mesh in enumerate(gltf.get("meshes", [])):
        for primitive_index, primitive in enumerate(mesh.get("primitives", [])):
            where = f"mesh {mesh_index} primitive {primitive_index}"
            attributes = primitive.get("attributes", {})
            if "POSITION" not in attributes:
                problems.append(f"{where} has no POSITION attribute")
                continue

            position_accessor = accessors[attributes["POSITION"]]
            if "min" not in position_accessor or "max" not in position_accessor:
                problems.append(f"{where} POSITION accessor must declare min and max")
            vertex_count = position_accessor["count"]

            for name, accessor_index in attributes.items():
                if accessors[accessor_index]["count"] != vertex_count:
                    problems.append(
                        f"{where} attribute {name} has "
                        f"{accessors[accessor_index]['count']} elements "
                        f"but POSITION has {vertex_count}"
                    )

            if "NORMAL" in attributes:
                normals = document.accessor(attributes["NORMAL"])
                lengths = np.linalg.norm(normals.astype(np.float64), axis=1)
                bad = int(np.count_nonzero(np.abs(lengths - 1.0) > UNIT_LENGTH_TOLERANCE))
                if bad:
                    problems.append(f"{where} has {bad} NORMAL vector(s) of non-unit length")

            has_joints = "JOINTS_0" in attributes
            has_weights = "WEIGHTS_0" in attributes
            if has_joints != has_weights:
                problems.append(f"{where} has JOINTS_0 without WEIGHTS_0 or the reverse")

            if has_joints:
                joints_accessor = accessors[attributes["JOINTS_0"]]
                if joints_accessor["componentType"] not in (5121, 5123):
                    problems.append(
                        f"{where} JOINTS_0 must be UNSIGNED_BYTE or UNSIGNED_SHORT"
                    )
                if joints_accessor["type"] != "VEC4":
                    problems.append(f"{where} JOINTS_0 must be VEC4")
                weights_accessor = accessors[attributes["WEIGHTS_0"]]
                if weights_accessor["type"] != "VEC4":
                    problems.append(f"{where} WEIGHTS_0 must be VEC4")

                weights = document.accessor(attributes["WEIGHTS_0"]).astype(np.float64)
                if np.any(weights < 0.0):
                    problems.append(f"{where} has negative WEIGHTS_0 values")
                sums = weights.sum(axis=1)
                unnormalised = int(
                    np.count_nonzero(np.abs(sums - 1.0) > WEIGHT_SUM_TOLERANCE)
                )
                if unnormalised:
                    worst = float(np.max(np.abs(sums - 1.0)))
                    problems.append(
                        f"{where} has {unnormalised} vertex weight set(s) that do not "
                        f"sum to 1 (worst drift {worst:.3e})"
                    )

                joints = document.accessor(attributes["JOINTS_0"]).astype(np.int64)
                zero_weight = weights == 0.0
                if np.any(joints[zero_weight] != 0):
                    problems.append(
                        f"{where} has a non-zero joint index on a zero-weight slot"
                    )

            if "indices" in primitive:
                indices_accessor = accessors[primitive["indices"]]
                if indices_accessor["type"] != "SCALAR":
                    problems.append(f"{where} indices accessor must be SCALAR")
                if indices_accessor["componentType"] not in (5121, 5123, 5125):
                    problems.append(f"{where} indices must be an unsigned integer type")
                if primitive.get("mode", 4) == 4 and indices_accessor["count"] % 3:
                    problems.append(
                        f"{where} has {indices_accessor['count']} indices, "
                        "not a multiple of 3 for TRIANGLES"
                    )
                indices = document.accessor(primitive["indices"])
                if len(indices) and int(indices.max()) >= vertex_count:
                    problems.append(
                        f"{where} index {int(indices.max())} is past the "
                        f"{vertex_count} vertices available"
                    )
    return problems


def _check_skins(gltf: dict, document: ParsedGLTF) -> list[str]:
    problems: list[str] = []
    nodes = gltf.get("nodes", [])
    accessors = gltf.get("accessors", [])

    parent_of: dict[int, int] = {}
    for index, node in enumerate(nodes):
        for child in node.get("children", []):
            parent_of[child] = index

    def ancestors(node_index: int) -> set[int]:
        chain = {node_index}
        cursor = parent_of.get(node_index)
        while cursor is not None and cursor not in chain:
            chain.add(cursor)
            cursor = parent_of.get(cursor)
        return chain

    for skin_index, skin in enumerate(gltf.get("skins", [])):
        joints = skin.get("joints", [])
        if not joints:
            problems.append(f"skin {skin_index} has no joints")
            continue
        for joint in joints:
            if joint >= len(nodes):
                problems.append(f"skin {skin_index} refers to missing node {joint}")

        if "inverseBindMatrices" in skin:
            accessor_index = skin["inverseBindMatrices"]
            accessor = accessors[accessor_index]
            if accessor["type"] != "MAT4" or accessor["componentType"] != 5126:
                problems.append(
                    f"skin {skin_index} inverseBindMatrices must be a float MAT4 accessor"
                )
            if accessor["count"] != len(joints):
                problems.append(
                    f"skin {skin_index} has {len(joints)} joints but "
                    f"{accessor['count']} inverse bind matrices"
                )
            view_index = accessor.get("bufferView")
            if view_index is not None:
                view = gltf["bufferViews"][view_index]
                # Spec: a bufferView used by inverseBindMatrices MUST NOT
                # define a target.
                if "target" in view:
                    problems.append(
                        f"skin {skin_index} inverseBindMatrices bufferView declares a target"
                    )
            matrices = document.accessor(accessor_index)
            for position, matrix in enumerate(matrices):
                if not np.all(np.isfinite(matrix)):
                    problems.append(
                        f"skin {skin_index} inverse bind matrix {position} is not finite"
                    )
                elif abs(float(np.linalg.det(matrix))) < 1e-12:
                    problems.append(
                        f"skin {skin_index} inverse bind matrix {position} is singular"
                    )

        if "skeleton" in skin:
            skeleton = skin["skeleton"]
            for joint in joints:
                if skeleton not in ancestors(joint):
                    problems.append(
                        f"skin {skin_index} skeleton node {skeleton} is not an "
                        f"ancestor of joint node {joint}"
                    )
                    break

    for node_index, node in enumerate(nodes):
        if "skin" in node and "mesh" not in node:
            problems.append(f"node {node_index} has a skin but no mesh")
        if "skin" in node:
            skin = gltf.get("skins", [])[node["skin"]]
            mesh = gltf["meshes"][node["mesh"]]
            joint_count = len(skin.get("joints", []))
            for primitive in mesh.get("primitives", []):
                attributes = primitive.get("attributes", {})
                if "JOINTS_0" not in attributes:
                    problems.append(
                        f"node {node_index} is skinned but its mesh has no JOINTS_0"
                    )
                    continue
                joints = document.accessor(attributes["JOINTS_0"]).astype(np.int64)
                if len(joints) and int(joints.max()) >= joint_count:
                    problems.append(
                        f"node {node_index} uses joint slot {int(joints.max())} but "
                        f"skin {node['skin']} only has {joint_count} joints"
                    )
    return problems


def _check_scenes(gltf: dict) -> list[str]:
    problems: list[str] = []
    scenes = gltf.get("scenes", [])
    nodes = gltf.get("nodes", [])
    if "scene" in gltf and gltf["scene"] >= len(scenes):
        problems.append(f"scene {gltf['scene']} does not exist")
    for index, scene in enumerate(scenes):
        for node in scene.get("nodes", []):
            if node >= len(nodes):
                problems.append(f"scene {index} refers to missing node {node}")
    return problems


def reachable_nodes(gltf: dict, scene_index: int = 0) -> set[int]:
    """
    Collect every node reachable from a scene.

    Used to prove that no bone branch was dropped from the hierarchy.

    Returns:
    - The set of reachable node indices.
    """
    nodes = gltf.get("nodes", [])
    scenes = gltf.get("scenes", [])
    if scene_index >= len(scenes):
        return set()

    reached: set[int] = set()
    stack = list(scenes[scene_index].get("nodes", []))
    while stack:
        current = stack.pop()
        if current in reached or current >= len(nodes):
            continue
        reached.add(current)
        stack.extend(nodes[current].get("children", []))
    return reached
