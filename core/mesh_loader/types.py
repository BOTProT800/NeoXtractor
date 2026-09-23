"""Types for Mesh Loader"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MAX_VERTEX_COUNT = 500000
MAX_FACE_COUNT = 250000
MAX_BONE_COUNT = 20000


@dataclass
class Mesh:
    """
    Represents the mesh part of a full mesh file.
    """

    vertexes: int = 0
    faces: int = 0
    position: list[tuple[float, float, float]] = field(
        default_factory=list[tuple[float, float, float]]
    )
    normal: list[tuple[float, float, float]] = field(
        default_factory=list[tuple[float, float, float]]
    )
    face: list[tuple[int, int, int]] = field(default_factory=list[tuple[int, int, int]])
    uv: list[tuple[float, float]] = field(default_factory=list[tuple[float, float]])


@dataclass
class Bones:
    """
    Represents the bone/skeleton part of a full mesh file.

    Matrix semantics
    ----------------
    ``matrix[i]`` holds the 16 floats found in the file, reshaped row-major into
    a ``4x4`` array. **No interpretation is applied at parse time**: the array is
    the raw storage, not a matrix in any particular mathematical convention.

    Deciding what those numbers mean is the job of
    :mod:`core.mesh_converter.skeleton`, which resolves two independent axes:

    * *storage* -- whether the file uses the row-vector layout (``v' = v @ M``,
      translation in the last **row**) or the column-vector layout
      (``v' = M @ v``, translation in the last **column**). This is detected
      from the data by looking at which of the two is the affine ``(0, 0, 0, 1)``
      slot, see :func:`core.mesh_converter.skeleton.detect_matrix_storage`.
    * *role* -- whether the matrix is the bone's global rest transform
      (bone space -> model space), its inverse, or a parent-relative transform.

    The NeoX variants seen so far store the **global rest transform in
    row-vector layout**, which is what the viewer
    (``gui/renderers/mesh_renderer.py``) and the IQE exporter already assume.
    That is the default, not a proven invariant for every NeoX game.

    ``joint_index_bits`` records the storage width of the per-vertex joint
    indices (8 or 16). It is what defines the sentinel value for an empty
    influence slot (``255`` or ``65535``); with 16-bit indices, ``255`` is an
    ordinary bone index and must not be treated as a sentinel.
    """

    # Bone/skeleton data
    has_bones: int = 0
    parents: list[int] = field(default_factory=list[int])
    names: list[str] = field(default_factory=list[str])
    matrix: list[np.ndarray] = field(default_factory=list[np.ndarray])
    count: int = 0

    # Storage width of the per-vertex joint indices as read from the file.
    joint_index_bits: int = 8

    # Vertex bone assignments
    joints: list[tuple[int, int, int, int]] = field(
        default_factory=list[tuple[int, int, int, int]]
    )
    weights: list[tuple[float, float, float, float]] = field(
        default_factory=list[tuple[float, float, float, float]]
    )

    @property
    def joint_index_sentinel(self) -> int:
        """Value that marks an unused influence slot for this index width."""
        return (1 << self.joint_index_bits) - 1


@dataclass
class MeshData:
    """
    Standardized mesh data structure containing all parsed mesh information.

    This dataclass provides a consistent interface for mesh data across all parsers,
    ensuring type safety and clear documentation of the expected data structure.
    """

    # Metadata
    version: int
    type: int
    mesh: Mesh = field(default_factory=Mesh)
    bones: Bones = field(default_factory=Bones)

    # Core mesh geometry

    @property
    def vertex_count(self) -> int:
        """Get the number of vertices in the mesh."""
        return self.mesh.vertexes

    @property
    def face_count(self) -> int:
        """Get the number of faces in the mesh."""
        return self.mesh.faces

    @property
    def uv_count(self) -> int:
        """Get the number of UV coordinates in the mesh."""
        return len(self.mesh.uv)

    @property
    def has_bones(self) -> bool:
        """Check if the mesh has bone data."""
        return self.bones.has_bones == 1 or self.bones.has_bones == 4

    @property
    def has_uvs(self) -> bool:
        """Check if the mesh has UV coordinate data."""
        return len(self.mesh.uv) == len(self.mesh.position)

    def validate_geometry(self) -> list[str]:
        """
        Check the geometry blocks for internal consistency.

        Geometry and rig are validated separately: a mesh can have perfectly
        usable geometry and an unusable rig, and the caller needs to tell those
        apart (a static export is still possible in that case).

        Returns:
            A list of human readable problems. Empty means the geometry is
            consistent.
        """
        problems: list[str] = []

        vertexes = self.mesh.vertexes
        if vertexes <= 0:
            problems.append(f"vertex count is {vertexes}, expected a positive value")
        if len(self.mesh.position) != vertexes:
            problems.append(
                f"position array holds {len(self.mesh.position)} entries "
                f"but the header declares {vertexes} vertices"
            )
        if self.mesh.normal and len(self.mesh.normal) != vertexes:
            problems.append(
                f"normal array holds {len(self.mesh.normal)} entries "
                f"but the header declares {vertexes} vertices"
            )
        if len(self.mesh.face) != self.mesh.faces:
            problems.append(
                f"face array holds {len(self.mesh.face)} entries "
                f"but the header declares {self.mesh.faces} faces"
            )

        # Face corners address *vertices*, not faces.
        vertex_limit = max(vertexes, len(self.mesh.position))
        for face_index, face in enumerate(self.mesh.face):
            if len(face) != 3:
                problems.append(f"face {face_index} has {len(face)} indices, expected 3")
                continue
            for corner in face:
                if corner < 0 or corner >= vertex_limit:
                    problems.append(
                        f"face {face_index} references vertex {corner}, "
                        f"outside of [0, {vertex_limit})"
                    )
                    break

        return problems

    def validate_rig(self) -> list[str]:
        """
        Check the skeleton and the per-vertex influences.

        Returns:
            A list of human readable problems. Empty means the rig is
            structurally usable; it does not mean the influences are non-empty.
        """
        problems: list[str] = []
        if not self.has_bones:
            return problems

        bones = self.bones
        bone_count = len(bones.names)

        if len(bones.parents) != bone_count:
            problems.append(
                f"parent array holds {len(bones.parents)} entries "
                f"but there are {bone_count} bone names"
            )
        if len(bones.matrix) != bone_count:
            problems.append(
                f"matrix array holds {len(bones.matrix)} entries "
                f"but there are {bone_count} bone names"
            )
        if bones.count != bone_count:
            problems.append(
                f"bone count field is {bones.count} "
                f"but there are {bone_count} bone names"
            )
        if bones.joint_index_bits not in (8, 16):
            problems.append(
                f"joint index width is {bones.joint_index_bits} bits, expected 8 or 16"
            )

        # Parent references, roots and cycles.
        parents_usable = len(bones.parents) == bone_count
        for index, parent in enumerate(bones.parents):
            if parent == -1:
                continue
            if parent < 0 or parent >= bone_count:
                problems.append(
                    f"bone {index} has parent {parent}, outside of [0, {bone_count})"
                )
                parents_usable = False
            elif parent == index:
                problems.append(f"bone {index} is its own parent")
                parents_usable = False
        if parents_usable and bones.parents:
            if -1 not in bones.parents:
                problems.append("skeleton has no root bone (no parent == -1)")
                parents_usable = False
        if parents_usable and bones.parents:
            for index in range(bone_count):
                seen: set[int] = set()
                cursor = index
                while cursor != -1:
                    if cursor in seen:
                        problems.append(
                            f"bone {index} sits on a parent cycle through bone {cursor}"
                        )
                        break
                    seen.add(cursor)
                    cursor = bones.parents[cursor]

        # Matrices must be usable as transforms.
        for index, matrix in enumerate(bones.matrix):
            array = np.asarray(matrix, dtype=np.float64)
            if array.shape != (4, 4):
                problems.append(
                    f"bone {index} matrix has shape {array.shape}, expected (4, 4)"
                )
                continue
            if not np.all(np.isfinite(array)):
                problems.append(f"bone {index} matrix holds non-finite values")
                continue
            if abs(float(np.linalg.det(array))) < 1e-12:
                problems.append(
                    f"bone {index} matrix is singular and cannot be inverted"
                )

        # Per-vertex influences.
        if len(bones.joints) != self.mesh.vertexes:
            problems.append(
                f"joint array holds {len(bones.joints)} entries "
                f"but the header declares {self.mesh.vertexes} vertices"
            )
        if len(bones.weights) != self.mesh.vertexes:
            problems.append(
                f"weight array holds {len(bones.weights)} entries "
                f"but the header declares {self.mesh.vertexes} vertices"
            )
        for vertex, weights in enumerate(bones.weights):
            for weight in weights:
                if not np.isfinite(weight):
                    problems.append(f"vertex {vertex} has a non-finite weight")
                    break
                if weight < 0.0:
                    problems.append(f"vertex {vertex} has a negative weight {weight}")
                    break

        return problems

    def validate(self) -> bool:
        """
        Validate the consistency of mesh data.

        Returns:
            True if both the geometry and the rig are consistent.
        """
        return not self.validate_geometry() and not self.validate_rig()


class BaseMeshParser(ABC):
    """Abstract base class for mesh parsers."""

    @abstractmethod
    def parse(self, data: bytes) -> MeshData:
        """
        Parse mesh data.

        Args:
            data: Raw mesh data as bytes

        Returns:
            MeshData object containing parsed mesh data

        Raises:
            MeshParsingError: If parsing fails
        """
        raise NotImplementedError

    def _standardize_mesh_data(self, model: dict[str, Any]) -> MeshData:
        """
        Convert raw parsed data to standardized MeshData object.

        Args:
            model: Raw parsed mesh data dictionary

        Returns:
            Standardized MeshData object with unified field names and structure
        """
        # Create MeshData with unified field mapping
        mesh_data = MeshData(
            # Metadata
            version=model["version"],
            type=model["type"],
            # Core mesh data
            mesh=Mesh(
                vertexes=model["mesh"]["data"][0],
                faces=model["mesh"]["data"][1],
                position=model["mesh"]["position"],
                normal=model["mesh"]["normal"],
                face=model["mesh"]["face"],
                uv=model["mesh"]["uv"],
            ),
            # Bone data
            bones=Bones(
                has_bones=model["bones"]["has_bones"],
                parents=model["bones"]["parent_connections"],
                names=model["bones"]["names"],
                matrix=model["bones"]["matrix"],
                count=model["bones"]["count"],
                joint_index_bits=model["bones"].get("joint_index_bits", 8),
                joints=model["bones"]["joints"],
                weights=model["bones"]["weights"],
            )
            if model["bones"]["has_bones"] == 1 or model["bones"]["has_bones"] == 4
            else Bones(
                has_bones=model["bones"]["has_bones"],
                parents=[],
                names=[],
                matrix=[],
                count=0,
                joints=[],
                weights=[],
            ),
        )
        return mesh_data

    def _validate_vertex_count(self, vertex_count: int) -> None:
        """
        Validate vertex count against maximum limits.

        Args:
            vertex_count: Number of vertices in the mesh

        Raises:
            ValueError: If vertex count exceeds maximum limit
        """
        if vertex_count == 0:
            raise ValueError("Vertex count cannot be zero")
        if vertex_count > MAX_VERTEX_COUNT:
            raise ValueError(
                f"Vertex count {vertex_count} exceeds maximum limit of {MAX_VERTEX_COUNT}"
            )

    def _validate_face_count(self, face_count: int) -> None:
        """
        Validate face count against maximum limits.

        Args:
            face_count: Number of faces in the mesh

        Raises:
            ValueError: If face count exceeds maximum limit
        """
        if face_count == 0:
            raise ValueError("Face count cannot be zero")
        if face_count > MAX_FACE_COUNT:
            raise ValueError(
                f"Face count {face_count} exceeds maximum limit of {MAX_FACE_COUNT}"
            )

    def _validate_bone_count(self, bone_count: int) -> None:
        """
        Validate bone count against maximum limits.

        Args:
            bone_count: Number of bones in the mesh

        Raises:
            ValueError: If bone count exceeds maximum limit
        """
        if bone_count > MAX_BONE_COUNT:
            raise ValueError(
                f"Bone count {bone_count} exceeds maximum limit of {MAX_BONE_COUNT}"
            )
