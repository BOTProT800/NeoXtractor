"""
Shared glTF 2.0 scene builder.

Both the ``.gltf`` and the ``.glb`` writers call :func:`build_scene`. It returns
the JSON structure plus the raw binary blob; how that blob is delivered (a
base64 data URI or a GLB ``BIN`` chunk) is the only difference between the two
formats, so the rig can never drift between them.

Scene layout
------------
::

    scene.nodes = [0, 1]
    node 0  "<name>"          mesh 0, skin 0 when the mesh is rigged
    node 1  "<name>Armature"  container, parent of every root bone
    node 2+ bones             node index == 2 + source bone index

The mesh node is a scene root on purpose: glTF ignores the transform of a
skinned mesh node, and a non-root skinned mesh makes validators warn about
parent transforms that will never apply. The armature container gives every
skeleton a single common root, so ``skin.skeleton`` stays valid whether the
file has one root bone or twelve.

Index spaces
------------
Three different index spaces meet here and used to be confused with each other:

* **source bone index** -- what the file stores in ``JOINTS`` and ``parents``.
* **joint slot** -- a position inside ``skin.joints``. This is what the
  ``JOINTS_0`` attribute holds, per the glTF specification.
* **node index** -- a position inside the top level ``nodes`` array.

``bone_to_slot`` and ``slot_to_node`` are built explicitly and carried on the
result so the mapping is inspectable instead of implied by an offset.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.logger import get_logger
from core.mesh_loader import MeshData
from core.mesh_converter.skeleton import (
    NEOX_TO_GLTF,
    CoordinateConversion,
    MatrixRole,
    MatrixStorage,
    Skeleton,
    SkeletonError,
    build_skeleton,
)

GENERATOR = "NeoXtractor glTF exporter"

# glTF component types.
COMPONENT_UNSIGNED_SHORT = 5123
COMPONENT_UNSIGNED_INT = 5125
COMPONENT_FLOAT = 5126

# glTF bufferView targets.
TARGET_ARRAY_BUFFER = 34962
TARGET_ELEMENT_ARRAY_BUFFER = 34963

MODE_TRIANGLES = 4

#: Influence slots per vertex in a single ``JOINTS_0``/``WEIGHTS_0`` pair.
INFLUENCES_PER_VERTEX = 4

#: How far a vertex's weights may sum away from 1.0 before the export is
#: refused. Storage is float32 and the files are already normalised, so a real
#: model lands within a few ULPs. A larger drift means the influences were not
#: read where they actually live, and renormalising would only hide that.
WEIGHT_SUM_TOLERANCE = 1e-2


class MeshExportError(ValueError):
    """The mesh cannot be exported as requested."""


class SkinDataError(MeshExportError):
    """The per-vertex influences are unusable and must not be guessed at."""


@dataclass
class GLTFScene:
    """A built glTF scene: JSON structure plus the binary blob it refers to."""

    json_data: dict[str, Any]
    binary: bytes
    skeleton: Skeleton | None = None
    bone_to_slot: dict[int, int] = field(default_factory=dict)
    slot_to_node: list[int] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)

    @property
    def is_skinned(self) -> bool:
        """True when the scene carries a skin, not merely a skeleton."""
        return "skins" in self.json_data


class _BufferBuilder:
    """Accumulates the binary blob and hands out bufferView indices."""

    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self._length = 0
        self.views: list[dict[str, Any]] = []

    def add_view(self, payload: bytes, target: int | None = None) -> int:
        """
        Append a block and return its bufferView index.

        Every block starts on a 4 byte boundary so accessors stay aligned to
        their component size, which the specification requires.
        """
        padding = (-self._length) % 4
        if padding:
            self._chunks.append(b"\x00" * padding)
            self._length += padding

        offset = self._length
        self._chunks.append(payload)
        self._length += len(payload)

        view: dict[str, Any] = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(payload),
        }
        if target is not None:
            view["target"] = target
        self.views.append(view)
        return len(self.views) - 1

    def build(self) -> bytes:
        """Return the accumulated blob."""
        return b"".join(self._chunks)


def _accessor(
    buffer_view: int,
    component_type: int,
    count: int,
    type_str: str,
    minimum: list[float] | None = None,
    maximum: list[float] | None = None,
) -> dict[str, Any]:
    """Build one accessor entry."""
    accessor: dict[str, Any] = {
        "bufferView": buffer_view,
        "componentType": component_type,
        "count": count,
        "type": type_str,
    }
    if minimum is not None:
        accessor["min"] = minimum
    if maximum is not None:
        accessor["max"] = maximum
    return accessor


def _pack(values, fmt: str) -> bytes:
    """Pack a flat sequence little-endian."""
    flat = list(values)
    return struct.pack(f"<{len(flat)}{fmt}", *flat)


def _serialize_column_major(matrix: np.ndarray) -> list[float]:
    """
    Flatten a column-vector matrix the way glTF stores matrices.

    glTF serialises in column-major order. For a column-vector matrix ``M``
    that is ``M.T`` read row by row. This is a serialisation step, not a change
    of interpretation -- the transpose in
    :func:`core.mesh_converter.skeleton.to_contract_matrix` is the other thing.
    """
    return [float(value) for value in np.asarray(matrix, dtype=np.float64).T.reshape(-1)]


def _prepare_geometry(
    mesh: MeshData, conversion: CoordinateConversion
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray, list[str]]:
    """
    Convert positions, normals, UVs and faces into export-ready arrays.

    Returns:
    - ``(positions, normals, uvs, faces, diagnostics)``. ``normals`` and
      ``uvs`` are None when the source does not provide usable ones.
    """
    diagnostics: list[str] = []
    problems = mesh.validate_geometry()
    if problems:
        raise MeshExportError(
            "mesh geometry is inconsistent: " + "; ".join(problems[:10])
        )

    vertex_count = mesh.mesh.vertexes
    basis = conversion.matrix[:3, :3]

    positions = np.asarray(mesh.mesh.position, dtype=np.float64)
    positions = positions @ basis.T + conversion.matrix[:3, 3]
    if not np.all(np.isfinite(positions)):
        raise MeshExportError("mesh positions contain non-finite values")

    normals: np.ndarray | None = None
    if mesh.mesh.normal:
        normals = np.asarray(mesh.mesh.normal, dtype=np.float64)
        normals = normals @ np.linalg.inv(basis)
        lengths = np.linalg.norm(normals, axis=1)
        degenerate = lengths < 1e-8
        degenerate_count = int(np.count_nonzero(degenerate))
        if degenerate_count:
            # glTF requires NORMAL to be unit length. A zero-length normal has
            # no direction to recover, so it is replaced and reported rather
            # than silently written out as an invalid vector.
            diagnostics.append(
                f"{degenerate_count} normal(s) had zero length and were replaced "
                "with (0, 0, 1) to satisfy the glTF unit-length requirement"
            )
            normals[degenerate] = np.array([0.0, 0.0, 1.0])
            lengths[degenerate] = 1.0
        normals = normals / lengths[:, None]

    uvs: np.ndarray | None = None
    if mesh.mesh.uv:
        raw_uv = np.asarray(mesh.mesh.uv, dtype=np.float64)
        if len(raw_uv) >= vertex_count:
            if len(raw_uv) > vertex_count:
                diagnostics.append(
                    f"UV array holds {len(raw_uv)} entries for {vertex_count} vertices; "
                    "exporting the first layer only"
                )
            uvs = raw_uv[:vertex_count]
        else:
            diagnostics.append(
                f"UV array holds only {len(raw_uv)} entries for {vertex_count} "
                "vertices; TEXCOORD_0 is omitted"
            )

    faces = np.asarray(mesh.mesh.face, dtype=np.uint32).reshape(-1, 3)
    if conversion.flips_winding:
        # Mirroring reverses the orientation of every triangle; swapping two
        # corners puts the front face back where it was.
        faces = faces[:, [0, 2, 1]]

    return positions, normals, uvs, faces, diagnostics


@dataclass
class _Influences:
    """Per-vertex joint slots and weights, ready to be written."""

    joints: np.ndarray
    weights: np.ndarray
    unskinned_vertices: list[int]
    diagnostics: list[str]


def _prepare_influences(
    mesh: MeshData, bone_to_slot: dict[int, int], joint_count: int
) -> _Influences:
    """
    Validate and normalise the per-vertex influences.

    What is allowed:
    - duplicate joints inside one vertex are merged,
    - weights are renormalised when they already sum to ~1,
    - an empty slot is written as joint 0 with weight 0.

    What is refused, because fixing it would hide a misread stream:
    - a positive weight on a sentinel or out-of-range joint,
    - negative or non-finite weights,
    - a weight sum that drifts further than :data:`WEIGHT_SUM_TOLERANCE`.
    """
    diagnostics: list[str] = []
    vertex_count = mesh.mesh.vertexes
    bones = mesh.bones
    sentinel = bones.joint_index_sentinel

    if len(bones.joints) != vertex_count or len(bones.weights) != vertex_count:
        raise SkinDataError(
            f"influence arrays hold {len(bones.joints)} joint and "
            f"{len(bones.weights)} weight entries for {vertex_count} vertices"
        )

    joints_out = np.zeros((vertex_count, INFLUENCES_PER_VERTEX), dtype=np.uint16)
    weights_out = np.zeros((vertex_count, INFLUENCES_PER_VERTEX), dtype=np.float32)
    unskinned: list[int] = []
    merged_vertices = 0
    worst_drift = 0.0

    for vertex in range(vertex_count):
        raw_joints = bones.joints[vertex]
        raw_weights = bones.weights[vertex]
        if len(raw_joints) != len(raw_weights):
            raise SkinDataError(
                f"vertex {vertex} has {len(raw_joints)} joints but "
                f"{len(raw_weights)} weights"
            )

        accumulated: dict[int, float] = {}
        for joint_index, weight in zip(raw_joints, raw_weights):
            weight = float(weight)
            if not np.isfinite(weight):
                raise SkinDataError(f"vertex {vertex} has a non-finite weight")
            if weight < 0.0:
                raise SkinDataError(
                    f"vertex {vertex} has a negative weight ({weight})"
                )
            if weight == 0.0:
                continue
            joint_index = int(joint_index)
            if joint_index == sentinel:
                raise SkinDataError(
                    f"vertex {vertex} puts weight {weight} on the empty-slot "
                    f"sentinel {sentinel} ({bones.joint_index_bits}-bit indices)"
                )
            if joint_index not in bone_to_slot:
                raise SkinDataError(
                    f"vertex {vertex} puts weight {weight} on bone {joint_index}, "
                    f"which is outside of [0, {joint_count})"
                )
            slot = bone_to_slot[joint_index]
            if slot in accumulated:
                merged_vertices += 1
            accumulated[slot] = accumulated.get(slot, 0.0) + weight

        if not accumulated:
            unskinned.append(vertex)
            continue

        total = sum(accumulated.values())
        drift = abs(total - 1.0)
        worst_drift = max(worst_drift, drift)
        if drift > WEIGHT_SUM_TOLERANCE:
            raise SkinDataError(
                f"vertex {vertex} weights sum to {total:.6f}, off by {drift:.6f}. "
                "That is far past a rounding error and points at influences read "
                "from the wrong offset; they are not renormalised."
            )

        ranked = sorted(accumulated.items(), key=lambda item: -item[1])
        if len(ranked) > INFLUENCES_PER_VERTEX:  # pragma: no cover - 4 inputs max
            raise SkinDataError(
                f"vertex {vertex} has {len(ranked)} distinct influences, "
                f"more than the {INFLUENCES_PER_VERTEX} a single set can hold"
            )

        for slot_position, (slot, weight) in enumerate(ranked):
            joints_out[vertex, slot_position] = slot
            weights_out[vertex, slot_position] = np.float32(weight / total)

        # Normalising in float64 and rounding to float32 can leave the stored
        # sum a few ULPs off 1.0, which glTF validators flag. Push the residue
        # onto the dominant influence so the written values sum exactly.
        stored_total = np.float32(weights_out[vertex].sum())
        residue = np.float32(1.0) - stored_total
        if residue != 0.0:
            weights_out[vertex, 0] = np.float32(weights_out[vertex, 0] + residue)

    if merged_vertices:
        diagnostics.append(
            f"{merged_vertices} duplicate joint reference(s) merged into a single "
            "influence"
        )
    if worst_drift > 0.0:
        diagnostics.append(
            f"largest weight-sum drift before normalisation: {worst_drift:.3e}"
        )

    return _Influences(joints_out, weights_out, unskinned, diagnostics)


def build_scene(
    mesh: MeshData,
    *,
    name: str = "NeoXMesh",
    conversion: CoordinateConversion = NEOX_TO_GLTF,
    matrix_storage: MatrixStorage = MatrixStorage.AUTO,
    matrix_role: MatrixRole = MatrixRole.GLOBAL_BIND,
    use_trs_nodes: bool = True,
) -> GLTFScene:
    """
    Build a glTF scene from parsed mesh data.

    Parameters:
    - mesh: the parsed mesh.
    - name: name given to the mesh node and the armature container.
    - conversion: basis change applied to geometry and skeleton alike.
    - matrix_storage: layout of the stored bone matrices, or AUTO to detect it.
    - matrix_role: what the stored bone matrices mean.
    - use_trs_nodes: write bone nodes as translation/rotation/scale, which is
      what glTF requires for animated nodes. Bones whose local transform TRS
      cannot reproduce fall back to a baked matrix regardless.

    Returns:
    - A :class:`GLTFScene` holding the JSON structure and the binary blob.

    Raises:
    - MeshExportError: geometry or skeleton cannot be exported.
    - SkinDataError: the influences are unusable.
    """
    logger = get_logger()
    diagnostics: list[str] = []

    positions, normals, uvs, faces, geometry_notes = _prepare_geometry(mesh, conversion)
    diagnostics.extend(geometry_notes)
    vertex_count = len(positions)

    buffers = _BufferBuilder()
    accessors: list[dict[str, Any]] = []

    # min/max are taken from the float32 values that actually get written, not
    # from the float64 originals: rounding could otherwise put the declared
    # bounds outside the stored data, which validators reject.
    stored_positions = positions.astype(np.float32)
    position_view = buffers.add_view(
        _pack(stored_positions.reshape(-1), "f"), TARGET_ARRAY_BUFFER
    )
    accessors.append(
        _accessor(
            position_view,
            COMPONENT_FLOAT,
            vertex_count,
            "VEC3",
            minimum=[float(value) for value in stored_positions.min(axis=0)],
            maximum=[float(value) for value in stored_positions.max(axis=0)],
        )
    )
    attributes: dict[str, int] = {"POSITION": 0}

    if normals is not None:
        normal_view = buffers.add_view(
            _pack(normals.astype(np.float32).reshape(-1), "f"), TARGET_ARRAY_BUFFER
        )
        attributes["NORMAL"] = len(accessors)
        accessors.append(_accessor(normal_view, COMPONENT_FLOAT, vertex_count, "VEC3"))

    if uvs is not None:
        uv_view = buffers.add_view(
            _pack(uvs.astype(np.float32).reshape(-1), "f"), TARGET_ARRAY_BUFFER
        )
        attributes["TEXCOORD_0"] = len(accessors)
        accessors.append(_accessor(uv_view, COMPONENT_FLOAT, vertex_count, "VEC2"))

    index_view = buffers.add_view(
        _pack(faces.astype(np.uint32).reshape(-1), "I"), TARGET_ELEMENT_ARRAY_BUFFER
    )
    index_accessor = len(accessors)
    accessors.append(
        _accessor(index_view, COMPONENT_UNSIGNED_INT, int(faces.size), "SCALAR")
    )

    mesh_node: dict[str, Any] = {"name": name, "mesh": 0}
    nodes: list[dict[str, Any]] = [mesh_node]
    scene_nodes = [0]

    gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": GENERATOR},
        "scene": 0,
        "scenes": [{"nodes": scene_nodes}],
        "nodes": nodes,
        "meshes": [
            {
                "name": name,
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": index_accessor,
                        "mode": MODE_TRIANGLES,
                    }
                ],
            }
        ],
        "accessors": accessors,
        "bufferViews": buffers.views,
    }

    skeleton: Skeleton | None = None
    bone_to_slot: dict[int, int] = {}
    slot_to_node: list[int] = []

    if mesh.has_bones and mesh.bones.names:
        rig_problems = mesh.validate_rig()
        if rig_problems:
            raise MeshExportError(
                "bone data is inconsistent: " + "; ".join(rig_problems[:10])
            )
        try:
            skeleton = build_skeleton(
                list(mesh.bones.parents),
                list(mesh.bones.names),
                list(mesh.bones.matrix),
                storage=matrix_storage,
                role=matrix_role,
                conversion=conversion,
                mesh_positions=mesh.mesh.position,
            )
        except SkeletonError as error:
            raise MeshExportError(f"skeleton cannot be built: {error}") from error
        diagnostics.extend(skeleton.diagnostics)

        # Nodes are created for every bone first; parents are linked in a
        # second pass. Linking while creating used to drop whole branches
        # whenever a parent appeared after its child in the file, which the
        # parser's own trailing `dummy_root` guarantees.
        armature_node_index = 1
        bone_node_base = 2

        # Joint slots follow the file's bone order, so a JOINTS value read from
        # the file is also its slot. The maps are kept anyway: a future
        # per-submesh bone palette only has to change what is built here.
        slot_to_bone: list[int] = [bone.source_index for bone in skeleton.bones]
        for slot, source_index in enumerate(slot_to_bone):
            bone_to_slot[source_index] = slot
            slot_to_node.append(bone_node_base + source_index)

        armature_node: dict[str, Any] = {"name": f"{name}Armature", "children": []}
        nodes.append(armature_node)
        scene_nodes.append(armature_node_index)

        for bone in skeleton.bones:
            node: dict[str, Any] = {"name": bone.name}
            if use_trs_nodes and bone.trs_exact:
                # TRS rather than a baked matrix: glTF only lets animation
                # channels target translation/rotation/scale.
                if any(abs(value) > 0.0 for value in bone.translation):
                    node["translation"] = list(bone.translation)
                if bone.rotation != (0.0, 0.0, 0.0, 1.0):
                    node["rotation"] = list(bone.rotation)
                if bone.scale != (1.0, 1.0, 1.0):
                    node["scale"] = list(bone.scale)
            elif not np.allclose(bone.local_rest, np.identity(4)):
                node["matrix"] = _serialize_column_major(bone.local_rest)
            nodes.append(node)

        for bone in skeleton.bones:
            node_index = bone_node_base + bone.source_index
            if bone.parent == -1:
                armature_node["children"].append(node_index)
            else:
                parent_node = nodes[bone_node_base + bone.parent]
                parent_node.setdefault("children", []).append(node_index)

        influences = _prepare_influences(mesh, bone_to_slot, len(skeleton.bones))
        diagnostics.extend(influences.diagnostics)

        if len(influences.unskinned_vertices) == vertex_count:
            # Bones but nothing to bind them to. Exporting a skin here would
            # advertise a rig that deforms nothing, so the skeleton ships on
            # its own and the caller is told why.
            diagnostics.append(
                "no vertex carries a usable influence; the skeleton is exported "
                "but no skin is attached"
            )
            logger.warning(
                "GLTF: mesh declares bones but no usable vertex influences; "
                "exporting the skeleton without a skin"
            )
        elif influences.unskinned_vertices:
            preview = ", ".join(
                str(index) for index in influences.unskinned_vertices[:8]
            )
            raise SkinDataError(
                f"{len(influences.unskinned_vertices)} vertices have no usable "
                f"influence while the rest do (first: {preview}). Binding them to "
                "the root would hide the real cause instead of fixing it."
            )
        else:
            # Inverse bind matrices go out in joint-slot order, the order the
            # specification indexes them by. They are derived from the already
            # converted global rest transforms, never converted a second time.
            inverse_bind_values: list[float] = []
            for source_index in slot_to_bone:
                inverse_bind_values.extend(
                    _serialize_column_major(skeleton.bones[source_index].inverse_bind)
                )
            inverse_bind_view = buffers.add_view(
                _pack(np.asarray(inverse_bind_values, dtype=np.float32), "f")
            )
            inverse_bind_accessor = len(accessors)
            accessors.append(
                _accessor(inverse_bind_view, COMPONENT_FLOAT, len(slot_to_bone), "MAT4")
            )

            joints_view = buffers.add_view(
                _pack(influences.joints.reshape(-1), "H"), TARGET_ARRAY_BUFFER
            )
            attributes["JOINTS_0"] = len(accessors)
            accessors.append(
                _accessor(
                    joints_view, COMPONENT_UNSIGNED_SHORT, vertex_count, "VEC4"
                )
            )
            weights_view = buffers.add_view(
                _pack(influences.weights.reshape(-1), "f"), TARGET_ARRAY_BUFFER
            )
            attributes["WEIGHTS_0"] = len(accessors)
            accessors.append(
                _accessor(weights_view, COMPONENT_FLOAT, vertex_count, "VEC4")
            )

            gltf["skins"] = [
                {
                    "name": f"{name}Skin",
                    "inverseBindMatrices": inverse_bind_accessor,
                    # skin.joints is indexed by JOINTS_0; every entry here is a
                    # node index, and slot order is the bone order of the file.
                    "joints": list(slot_to_node),
                    # The armature container is a common root of every joint,
                    # which is what the specification asks of skeleton.
                    "skeleton": armature_node_index,
                }
            ]
            mesh_node["skin"] = 0

    for note in diagnostics:
        logger.info("GLTF: %s", note)

    return GLTFScene(
        json_data=gltf,
        binary=buffers.build(),
        skeleton=skeleton,
        bone_to_slot=bone_to_slot,
        slot_to_node=slot_to_node,
        diagnostics=diagnostics,
    )
