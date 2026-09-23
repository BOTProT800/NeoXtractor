"""
CPU skinning evaluated from an exported glTF/GLB file.

Nothing here imports the exporter. Node transforms, joint matrices and the
weighted vertex blend are all rebuilt from the file's own JSON and accessors,
following the skinning equation in the glTF specification::

    jointMatrix[j] = inverse(globalTransform(meshNode))
                     * globalTransform(joints[j])
                     * inverseBindMatrix[j]

    skinned = sum_i  weight[i] * jointMatrix[joints[i]] * position

That is what an importer does, so agreeing with it is evidence the file says
what the exporter meant, not merely that the exporter is self-consistent.
"""

from __future__ import annotations

import numpy as np

from .gltf_reader import ParsedGLTF


def quaternion_to_matrix(quaternion) -> np.ndarray:
    """Convert an ``(x, y, z, w)`` quaternion to a 3x3 rotation matrix."""
    x, y, z, w = (float(value) for value in quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def node_local_matrix(node: dict) -> np.ndarray:
    """
    Build a node's local transform from its glTF properties.

    A ``matrix`` property wins; otherwise the TRS triple is composed as
    ``T * R * S``, the order the specification mandates.
    """
    if "matrix" in node:
        # glTF stores matrices column-major.
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T

    translation = np.asarray(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    rotation = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    scale = np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)

    matrix = np.identity(4)
    matrix[:3, :3] = quaternion_to_matrix(rotation) * scale
    matrix[:3, 3] = translation
    return matrix


def global_node_matrices(
    gltf: dict, overrides: dict[int, np.ndarray] | None = None
) -> list[np.ndarray]:
    """
    Resolve every node's global transform.

    Parameters:
    - gltf: the document's JSON structure.
    - overrides: node index -> replacement local transform, to pose the rig.

    Returns:
    - One 4x4 global transform per node.
    """
    nodes = gltf.get("nodes", [])
    overrides = overrides or {}
    globals_: list[np.ndarray | None] = [None] * len(nodes)

    parent_of: dict[int, int] = {}
    for index, node in enumerate(nodes):
        for child in node.get("children", []):
            parent_of[child] = index

    def resolve(index: int) -> np.ndarray:
        cached = globals_[index]
        if cached is not None:
            return cached
        local = overrides.get(index)
        if local is None:
            local = node_local_matrix(nodes[index])
        parent = parent_of.get(index)
        result = local if parent is None else resolve(parent) @ local
        globals_[index] = result
        return result

    return [resolve(index) for index in range(len(nodes))]


def find_skinned_mesh_node(gltf: dict) -> int | None:
    """Return the index of the first node carrying both a mesh and a skin."""
    for index, node in enumerate(gltf.get("nodes", [])):
        if "mesh" in node and "skin" in node:
            return index
    return None


def skin_vertices(
    document: ParsedGLTF,
    overrides: dict[int, np.ndarray] | None = None,
    primitive_index: int = 0,
) -> np.ndarray:
    """
    Evaluate the skinned positions of a document's first skinned primitive.

    Parameters:
    - document: a parsed glTF or GLB file.
    - overrides: node index -> replacement local transform, to pose the rig.
    - primitive_index: which primitive of the skinned mesh to evaluate.

    Returns:
    - ``(vertex_count, 3)`` positions in the scene's space.

    Raises:
    - ValueError: the document has no skinned mesh.
    """
    gltf = document.json_data
    mesh_node_index = find_skinned_mesh_node(gltf)
    if mesh_node_index is None:
        raise ValueError("this document has no skinned mesh node")

    node = gltf["nodes"][mesh_node_index]
    skin = gltf["skins"][node["skin"]]
    primitive = gltf["meshes"][node["mesh"]]["primitives"][primitive_index]
    attributes = primitive["attributes"]

    positions = document.accessor(attributes["POSITION"]).astype(np.float64)
    joints = document.accessor(attributes["JOINTS_0"]).astype(np.int64)
    weights = document.accessor(attributes["WEIGHTS_0"]).astype(np.float64)

    joint_nodes = skin["joints"]
    if "inverseBindMatrices" in skin:
        inverse_bind = document.accessor(skin["inverseBindMatrices"])
    else:
        # The specification says a missing accessor means identity matrices.
        inverse_bind = np.array([np.identity(4) for _ in joint_nodes])

    globals_ = global_node_matrices(gltf, overrides)
    mesh_inverse = np.linalg.inv(globals_[mesh_node_index])
    joint_matrices = [
        mesh_inverse @ globals_[joint_nodes[slot]] @ inverse_bind[slot]
        for slot in range(len(joint_nodes))
    ]

    result = np.zeros_like(positions)
    for vertex in range(len(positions)):
        homogeneous = np.array([*positions[vertex], 1.0])
        blended = np.zeros(4)
        for influence in range(weights.shape[1]):
            weight = weights[vertex, influence]
            if weight == 0.0:
                continue
            blended += weight * (joint_matrices[joints[vertex, influence]] @ homogeneous)
        result[vertex] = blended[:3]
    return result


def rotation_override(
    node: dict, angle_degrees: float, axis: str = "z"
) -> np.ndarray:
    """
    Build a posed local transform: the node's rest transform, then a rotation.

    The rotation is applied in the node's own space, so the bone turns around
    its own pivot -- which is exactly what has to be verified.

    Returns:
    - The posed local transform.
    """
    angle = np.radians(angle_degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    if axis == "x":
        basis = np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]], dtype=np.float64)
    elif axis == "y":
        basis = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]], dtype=np.float64)
    else:
        basis = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float64)

    rotation = np.identity(4)
    rotation[:3, :3] = basis
    return node_local_matrix(node) @ rotation
