"""
Normalised skeleton contract shared by the mesh exporters.

Why this module exists
----------------------
``Bones.matrix`` is 16 raw floats per bone. Nothing in the file says whether
they are a global transform, its inverse, or a parent-relative one, nor whether
the layout is row-vector or column-vector. Every exporter used to re-invent an
answer, and they did not agree with each other. This module answers it once.

The internal contract
---------------------
Column vectors and ``4x4`` matrices throughout::

    G[j] = global rest transform of bone j   (bone space -> model space)
    L[j] = inverse(G[parent[j]]) @ G[j]      (root: L[j] = G[j])
    B[j] = inverse(G[j])                     (the glTF inverse bind matrix)
    S[j, pose] = G[j, pose] @ B[j]
    v[pose] = sum_j  weight[v, j] * S[j, pose] @ v[rest]

At rest ``G[j, pose] == G[j]`` so ``S[j] == identity``, which is necessary but
**not** sufficient: computing a matrix and its inverse from the same wrong
interpretation also passes that test. Pivots and orientations have to be
checked against something independent, which is why the raw matrix and the
source name are kept on every bone.

Coordinate conversion
---------------------
A basis change ``C`` is applied consistently:

* positions      -> ``C @ p``
* transforms     -> ``C @ M @ inverse(C)``
* normals        -> ``inverse(C).T @ n`` (equal to ``C`` for the mirrors here)
* triangle order -> reversed when ``det(C) < 0``

Because ``B[j]`` is derived from the *converted* ``G[j]``, the inverse bind
matrices follow automatically; they are never converted a second time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# Below this, a matrix is treated as singular and cannot be inverted.
SINGULAR_DETERMINANT = 1e-12

# A basis vector shorter than this is treated as a collapsed axis.
MIN_AXIS_LENGTH = 1e-9

# Largest element-wise error tolerated when a decomposed TRS is recomposed.
TRS_RECOMPOSE_TOLERANCE = 1e-5

# Element-wise tolerance when deciding which slot carries the affine row.
AFFINE_SLOT_TOLERANCE = 1e-4


class SkeletonError(ValueError):
    """The bone data cannot be turned into a usable skeleton."""


class MatrixStorage(Enum):
    """How the 16 stored floats map onto a transform."""

    #: ``v' = v @ M``; translation sits in the last row, last column is
    #: ``(0, 0, 0, 1)``. Contract matrix is the transpose of the stored array.
    ROW_VECTOR = "row_vector"

    #: ``v' = M @ v``; translation sits in the last column, last row is
    #: ``(0, 0, 0, 1)``. Contract matrix is the stored array unchanged.
    COLUMN_VECTOR = "column_vector"

    #: Decide from the data, see :func:`detect_matrix_storage`.
    AUTO = "auto"


class MatrixRole(Enum):
    """What the stored transform means once the layout is resolved."""

    #: Bone space -> model space, absolute. The NeoX default.
    GLOBAL_BIND = "global_bind"

    #: Model space -> bone space, i.e. already the inverse bind matrix.
    INVERSE_BIND = "inverse_bind"

    #: Bone space -> parent bone space.
    LOCAL_BIND = "local_bind"


@dataclass(frozen=True)
class CoordinateConversion:
    """A named basis change applied to the whole export."""

    name: str
    matrix: np.ndarray

    @property
    def flips_winding(self) -> bool:
        """True when the conversion mirrors, so triangle order must flip."""
        return bool(np.linalg.det(self.matrix[:3, :3]) < 0.0)

    @property
    def inverse(self) -> np.ndarray:
        """Inverse of the basis change."""
        return np.linalg.inv(self.matrix)

    def point(self, vector) -> np.ndarray:
        """Convert a position."""
        array = np.asarray(vector, dtype=np.float64)
        return self.matrix[:3, :3] @ array + self.matrix[:3, 3]

    def direction(self, vector) -> np.ndarray:
        """Convert a normal or any other direction."""
        array = np.asarray(vector, dtype=np.float64)
        return np.linalg.inv(self.matrix[:3, :3]).T @ array

    def transform(self, matrix) -> np.ndarray:
        """Convert a transform: ``C @ M @ inverse(C)``."""
        array = np.asarray(matrix, dtype=np.float64)
        return self.matrix @ array @ self.inverse


#: No conversion; export exactly the source basis.
IDENTITY_CONVERSION = CoordinateConversion("identity", np.identity(4))

#: Mirror X. This is the conversion the viewer (``mesh_renderer``) and the IQE
#: exporter already apply, and it is what turns the left-handed NeoX basis into
#: the right-handed basis glTF requires. Mirroring also reverses triangle
#: winding, which this module reports through ``flips_winding``.
NEOX_TO_GLTF = CoordinateConversion(
    "neox_flip_x", np.diag(np.array([-1.0, 1.0, 1.0, 1.0]))
)


@dataclass
class SkeletonBone:
    """One bone expressed in the internal contract."""

    #: Index of the bone in the source arrays. Preserved so joint indices read
    #: from the file keep meaning, and so animation clips can be matched later.
    source_index: int

    #: Name exactly as stored in the file.
    name: str

    #: Source index of the parent, ``-1`` for a root.
    parent: int

    #: Source indices of the direct children, in source order.
    children: list[int] = field(default_factory=list)

    #: ``G[j]``: global rest transform, after coordinate conversion.
    global_rest: np.ndarray = field(default_factory=lambda: np.identity(4))

    #: ``L[j]``: rest transform relative to the parent.
    local_rest: np.ndarray = field(default_factory=lambda: np.identity(4))

    #: ``B[j] = inverse(G[j])``: the glTF inverse bind matrix.
    inverse_bind: np.ndarray = field(default_factory=lambda: np.identity(4))

    #: Local TRS, ready for animation channels. ``rotation`` is ``(x, y, z, w)``.
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    #: Largest element-wise error when the TRS above is recomposed into
    #: ``local_rest``. Above ``TRS_RECOMPOSE_TOLERANCE`` the matrix carries
    #: shear or some other component TRS cannot express.
    trs_error: float = 0.0

    #: Raw array as found in the file, untouched, for diagnostics.
    source_matrix: np.ndarray = field(default_factory=lambda: np.identity(4))

    @property
    def trs_exact(self) -> bool:
        """True when the TRS triple reproduces ``local_rest``."""
        return self.trs_error <= TRS_RECOMPOSE_TOLERANCE

    @property
    def is_root(self) -> bool:
        """True when the bone has no parent."""
        return self.parent == -1


@dataclass
class Skeleton:
    """A validated hierarchy plus the conventions used to build it."""

    bones: list[SkeletonBone]
    roots: list[int]
    order: list[int]
    storage: MatrixStorage
    role: MatrixRole
    conversion: CoordinateConversion
    diagnostics: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.bones)

    @property
    def names(self) -> list[str]:
        """Bone names in source order."""
        return [bone.name for bone in self.bones]

    def global_rest_matrices(self) -> list[np.ndarray]:
        """``G[j]`` for every bone, in source order."""
        return [bone.global_rest for bone in self.bones]

    def inverse_bind_matrices(self) -> list[np.ndarray]:
        """``B[j]`` for every bone, in source order."""
        return [bone.inverse_bind for bone in self.bones]

    def pose_globals(self, local_overrides: dict[int, np.ndarray]) -> list[np.ndarray]:
        """
        Evaluate ``G[j, pose]`` with some local transforms replaced.

        Parameters:
        - local_overrides: source bone index -> replacement local transform.
          Bones left out keep their rest local transform.

        Returns:
        - The posed global transform of every bone, in source order.
        """
        globals_: list[np.ndarray] = [np.identity(4) for _ in self.bones]
        for index in self.order:
            bone = self.bones[index]
            local = np.asarray(
                local_overrides.get(index, bone.local_rest), dtype=np.float64
            )
            if bone.parent == -1:
                globals_[index] = local
            else:
                globals_[index] = globals_[bone.parent] @ local
        return globals_

    def skinning_matrices(
        self, local_overrides: dict[int, np.ndarray] | None = None
    ) -> list[np.ndarray]:
        """``S[j] = G[j, pose] @ B[j]`` for every bone, in source order."""
        globals_ = self.pose_globals(local_overrides or {})
        return [
            globals_[index] @ bone.inverse_bind
            for index, bone in enumerate(self.bones)
        ]


def quaternion_from_matrix(rotation) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a unit quaternion.

    Parameters:
    - rotation: orthonormal 3x3 matrix with positive determinant.

    Returns:
    - ``(x, y, z, w)`` as float64, the order glTF stores rotations in.
    """
    matrix = np.asarray(rotation, dtype=np.float64)
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m21 - m12) / scale
        y = (m02 - m20) / scale
        z = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / scale
        x = 0.25 * scale
        y = (m01 + m10) / scale
        z = (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / scale
        x = (m01 + m10) / scale
        y = 0.25 * scale
        z = (m12 + m21) / scale
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / scale
        x = (m02 + m20) / scale
        y = (m12 + m21) / scale
        z = 0.25 * scale

    quaternion = np.array([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm < MIN_AXIS_LENGTH:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return quaternion / norm


def matrix_from_quaternion(quaternion) -> np.ndarray:
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


def euler_xyz_from_matrix(matrix) -> np.ndarray:
    """
    Extract XYZ Euler angles, in radians, from a transform.

    The angles reproduce the rotation as ``Rz @ Ry @ Rx``: rotate about X
    first, then Y, then Z. That is the order Valve's SMD uses, and the one
    Blender calls ``'XYZ'``.

    Parameters:
    - matrix: 4x4 or 3x3 transform. Any scale is normalised away first.

    Returns:
    - ``(rx, ry, rz)`` as float64.
    """
    array = np.asarray(matrix, dtype=np.float64)
    basis = array[:3, :3].copy()

    scale = np.linalg.norm(basis, axis=0)
    if float(np.linalg.det(basis)) < 0.0:
        scale[0] = -scale[0]
    safe_scale = np.where(np.abs(scale) < MIN_AXIS_LENGTH, 1.0, scale)
    basis = basis / safe_scale

    # cos(ry), recovered from the part of the first column Y does not touch.
    cos_y = math.sqrt(basis[0, 0] * basis[0, 0] + basis[1, 0] * basis[1, 0])
    if cos_y > MIN_AXIS_LENGTH:
        rx = math.atan2(basis[2, 1], basis[2, 2])
        ry = math.atan2(-basis[2, 0], cos_y)
        rz = math.atan2(basis[1, 0], basis[0, 0])
    else:
        # Gimbal lock: Y is at +-90 degrees and X and Z become the same axis,
        # so the split between them is arbitrary. Put it all on X.
        rx = math.atan2(-basis[1, 2], basis[1, 1])
        ry = math.atan2(-basis[2, 0], cos_y)
        rz = 0.0
    return np.array([rx, ry, rz], dtype=np.float64)


def compose_trs(translation, rotation, scale) -> np.ndarray:
    """Rebuild a 4x4 matrix from translation, ``(x, y, z, w)`` rotation, scale."""
    matrix = np.identity(4)
    matrix[:3, :3] = matrix_from_quaternion(rotation) * np.asarray(
        scale, dtype=np.float64
    )
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def decompose_trs(
    matrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Split an affine matrix into translation, rotation and scale.

    glTF requires animated nodes to use TRS rather than a baked matrix, so
    every bone gets decomposed even when the exporter writes the matrix form.

    Parameters:
    - matrix: 4x4 column-vector transform.

    Returns:
    - ``(translation, rotation_xyzw, scale, error)`` where ``error`` is the
      largest element-wise difference between the recomposed matrix and the
      input. A non-zero error means the matrix carries shear or something else
      TRS cannot express; the caller decides what to do about it.
    """
    array = np.asarray(matrix, dtype=np.float64)
    translation = array[:3, 3].copy()
    basis = array[:3, :3].copy()

    scale = np.linalg.norm(basis, axis=0)
    if float(np.linalg.det(basis)) < 0.0:
        # A mirrored basis cannot be expressed by a rotation alone; fold the
        # reflection into the first axis rather than losing it.
        scale[0] = -scale[0]

    safe_scale = np.where(np.abs(scale) < MIN_AXIS_LENGTH, 1.0, scale)
    rotation_basis = basis / safe_scale
    rotation = quaternion_from_matrix(rotation_basis)

    recomposed = compose_trs(translation, rotation, scale)
    error = float(np.max(np.abs(recomposed - array)))
    return translation, rotation, scale, error


def detect_matrix_storage(
    matrices, tolerance: float = AFFINE_SLOT_TOLERANCE
) -> tuple[MatrixStorage, str]:
    """
    Decide the storage layout from the numbers themselves.

    An affine transform has ``(0, 0, 0, 1)`` in exactly one of two places: the
    last row (column-vector layout) or the last column (row-vector layout).
    Checking which one holds is a direct measurement, not a guess.

    Parameters:
    - matrices: iterable of 4x4 arrays as stored in the file.
    - tolerance: element-wise tolerance for the affine slot.

    Returns:
    - ``(storage, reason)``. When both slots qualify -- every matrix is a pure
      rotation around the origin -- the layouts are indistinguishable and the
      row-vector default is reported as such.
    """
    affine = np.array([0.0, 0.0, 0.0, 1.0])
    last_row_affine = True
    last_column_affine = True
    count = 0

    for matrix in matrices:
        array = np.asarray(matrix, dtype=np.float64)
        if array.shape != (4, 4):
            continue
        count += 1
        if not np.allclose(array[3, :], affine, atol=tolerance):
            last_row_affine = False
        if not np.allclose(array[:, 3], affine, atol=tolerance):
            last_column_affine = False

    if count == 0:
        return MatrixStorage.ROW_VECTOR, "no matrices to inspect, assuming row-vector"
    if last_column_affine and not last_row_affine:
        return (
            MatrixStorage.ROW_VECTOR,
            "last column is (0,0,0,1) in every matrix: translation lives in the last row",
        )
    if last_row_affine and not last_column_affine:
        return (
            MatrixStorage.COLUMN_VECTOR,
            "last row is (0,0,0,1) in every matrix: translation lives in the last column",
        )
    if last_row_affine and last_column_affine:
        return (
            MatrixStorage.ROW_VECTOR,
            "matrices carry no translation, layouts are indistinguishable; "
            "assuming row-vector to match the viewer and the IQE exporter",
        )
    return (
        MatrixStorage.ROW_VECTOR,
        "neither slot is affine, the data may not be plain transforms; "
        "assuming row-vector to match the viewer and the IQE exporter",
    )


def to_contract_matrix(matrix, storage: MatrixStorage) -> np.ndarray:
    """Reinterpret a stored array as a column-vector transform."""
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (4, 4):
        raise SkeletonError(f"bone matrix has shape {array.shape}, expected (4, 4)")
    if storage is MatrixStorage.ROW_VECTOR:
        # Transposing changes the interpretation. It is not the same operation
        # as serialising in column-major order, which happens at write time.
        return array.T.copy()
    return array.copy()


def _invert(matrix: np.ndarray, index: int, what: str) -> np.ndarray:
    determinant = float(np.linalg.det(matrix))
    if not math.isfinite(determinant) or abs(determinant) < SINGULAR_DETERMINANT:
        raise SkeletonError(
            f"bone {index}: {what} is singular (determinant {determinant:g}) "
            "and cannot be inverted"
        )
    return np.linalg.inv(matrix)


def _topological_order(parents: list[int]) -> list[int]:
    """Return bone indices ordered parents-before-children, or raise on a cycle."""
    order: list[int] = []
    state = [0] * len(parents)  # 0 unvisited, 1 in progress, 2 done

    for start in range(len(parents)):
        if state[start] == 2:
            continue
        chain: list[int] = []
        cursor = start
        while cursor != -1 and state[cursor] == 0:
            state[cursor] = 1
            chain.append(cursor)
            cursor = parents[cursor]
        if cursor != -1 and state[cursor] == 1:
            raise SkeletonError(
                f"bone {start} sits on a parent cycle through bone {cursor}"
            )
        for index in reversed(chain):
            state[index] = 2
            order.append(index)
    return order


def build_skeleton(
    parents: list[int],
    names: list[str],
    matrices: list[np.ndarray],
    *,
    storage: MatrixStorage = MatrixStorage.AUTO,
    role: MatrixRole = MatrixRole.GLOBAL_BIND,
    conversion: CoordinateConversion = NEOX_TO_GLTF,
    mesh_positions: list | None = None,
) -> Skeleton:
    """
    Turn raw bone arrays into the internal contract.

    Parameters:
    - parents: source index of each bone's parent, ``-1`` for a root.
    - names: bone names, same length as ``parents``.
    - matrices: 4x4 arrays exactly as stored in the file.
    - storage: layout of the stored matrices, or ``AUTO`` to detect it.
    - role: what the stored transform means.
    - conversion: basis change applied to the whole skeleton.
    - mesh_positions: optional rest positions, used only to flag a skeleton
      whose bone origins land nowhere near the geometry.

    Returns:
    - A :class:`Skeleton` with ``G``, ``L``, ``B`` and a local TRS per bone.

    Raises:
    - SkeletonError: inconsistent arrays, out-of-range parents, cycles, or a
      matrix that cannot be inverted.
    """
    count = len(parents)
    if len(names) != count:
        raise SkeletonError(
            f"{len(names)} bone names for {count} parents; the arrays must match"
        )
    if len(matrices) != count:
        raise SkeletonError(
            f"{len(matrices)} bone matrices for {count} parents; the arrays must match"
        )
    for index, parent in enumerate(parents):
        if parent == -1:
            continue
        if parent < 0 or parent >= count:
            raise SkeletonError(
                f"bone {index} has parent {parent}, outside of [0, {count})"
            )
        if parent == index:
            raise SkeletonError(f"bone {index} is its own parent")

    diagnostics: list[str] = []

    resolved_storage = storage
    if storage is MatrixStorage.AUTO:
        resolved_storage, reason = detect_matrix_storage(matrices)
        diagnostics.append(f"matrix storage detected as {resolved_storage.value}: {reason}")
    else:
        diagnostics.append(f"matrix storage declared as {resolved_storage.value}")
    diagnostics.append(f"matrix role declared as {role.value}")
    diagnostics.append(f"coordinate conversion: {conversion.name}")

    order = _topological_order(parents)

    contract = [to_contract_matrix(matrix, resolved_storage) for matrix in matrices]

    # Resolve the role into absolute rest transforms.
    if role is MatrixRole.GLOBAL_BIND:
        globals_ = contract
    elif role is MatrixRole.INVERSE_BIND:
        globals_ = [
            _invert(matrix, index, "inverse bind matrix")
            for index, matrix in enumerate(contract)
        ]
    elif role is MatrixRole.LOCAL_BIND:
        globals_ = [np.identity(4) for _ in range(count)]
        for index in order:
            parent = parents[index]
            globals_[index] = (
                contract[index]
                if parent == -1
                else globals_[parent] @ contract[index]
            )
    else:  # pragma: no cover - the enum has no other members
        raise SkeletonError(f"unsupported matrix role {role}")

    # Apply the basis change to the absolute transforms only. Deriving L and B
    # from the converted G keeps every derived matrix consistent by
    # construction instead of converting each one separately.
    globals_ = [conversion.transform(matrix) for matrix in globals_]

    bones: list[SkeletonBone] = []
    for index in range(count):
        bones.append(
            SkeletonBone(
                source_index=index,
                name=names[index],
                parent=parents[index],
                global_rest=globals_[index],
                source_matrix=np.asarray(matrices[index], dtype=np.float64).copy(),
            )
        )

    for index in range(count):
        parent = parents[index]
        if parent != -1:
            bones[parent].children.append(index)

    shear_bones: list[int] = []
    for index in order:
        bone = bones[index]
        if bone.parent == -1:
            local = bone.global_rest
        else:
            parent_inverse = _invert(
                globals_[bone.parent], bone.parent, "global rest transform"
            )
            local = parent_inverse @ bone.global_rest
        bone.local_rest = local
        bone.inverse_bind = _invert(bone.global_rest, index, "global rest transform")

        translation, rotation, scale, error = decompose_trs(local)
        bone.translation = tuple(float(value) for value in translation)
        bone.rotation = tuple(float(value) for value in rotation)
        bone.scale = tuple(float(value) for value in scale)
        bone.trs_error = error
        if error > TRS_RECOMPOSE_TOLERANCE:
            shear_bones.append(index)

    if shear_bones:
        preview = ", ".join(str(index) for index in shear_bones[:8])
        suffix = "" if len(shear_bones) <= 8 else f" (+{len(shear_bones) - 8} more)"
        diagnostics.append(
            f"{len(shear_bones)} bone(s) have a local transform TRS cannot reproduce "
            f"(shear or a degenerate axis): {preview}{suffix}. "
            "Those nodes are written as a baked matrix and cannot be animated as-is."
        )

    roots = [index for index in range(count) if parents[index] == -1]
    if not roots and count:
        raise SkeletonError("skeleton has no root bone (no parent == -1)")
    if len(roots) > 1:
        diagnostics.append(
            f"{len(roots)} root bones; they are parented to the export container node "
            "and keep their original bone indices"
        )

    if mesh_positions is not None and len(mesh_positions) and count:
        diagnostics.extend(_role_sanity_diagnostics(globals_, mesh_positions, conversion))

    return Skeleton(
        bones=bones,
        roots=roots,
        order=order,
        storage=resolved_storage,
        role=role,
        conversion=conversion,
        diagnostics=diagnostics,
    )


def _role_sanity_diagnostics(
    globals_: list[np.ndarray],
    mesh_positions: list,
    conversion: CoordinateConversion,
) -> list[str]:
    """
    Flag a skeleton whose bone origins sit nowhere near the geometry.

    This does not prove the role is wrong -- a rig legitimately has bones
    outside the mesh bounds -- but reading a global transform as an inverse
    bind matrix (or the other way round) usually throws the origins far out,
    and that is worth saying out loud rather than silently exporting it.
    """
    positions = np.asarray(mesh_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        return []
    converted = positions @ conversion.matrix[:3, :3].T + conversion.matrix[:3, 3]

    mesh_min = converted.min(axis=0)
    mesh_max = converted.max(axis=0)
    mesh_center = (mesh_min + mesh_max) * 0.5
    mesh_radius = float(np.linalg.norm(mesh_max - mesh_min)) * 0.5
    if mesh_radius < MIN_AXIS_LENGTH:
        return []

    origins = np.array([matrix[:3, 3] for matrix in globals_], dtype=np.float64)
    distances = np.linalg.norm(origins - mesh_center, axis=1)
    outliers = int(np.count_nonzero(distances > mesh_radius * 10.0))
    if outliers:
        return [
            f"{outliers} of {len(origins)} bone origins sit more than 10x the mesh "
            "radius away from its centre. Check the declared matrix role and layout; "
            "the geometry and the skeleton may be in different spaces."
        ]
    return []
