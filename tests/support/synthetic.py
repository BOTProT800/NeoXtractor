"""
Synthetic fixtures for the mesh export pipeline.

Two kinds of fixture live here:

* :func:`build_mesh_file` writes a byte-exact ``.mesh`` blob that the real
  :class:`~core.mesh_loader.parsers.new_parser.MeshParser0` parses. It is the
  only way to test the binary reader, including how many bytes each block
  consumes.
* :func:`make_mesh_data` builds a :class:`~core.mesh_loader.types.MeshData`
  directly, for exporter tests that do not need to go through the reader.

The file layout below was read off the parser; every offset is written
explicitly so a change in either side shows up as a failing test rather than a
silently shifted block.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from core.mesh_loader.types import Bones, Mesh, MeshData

MAGIC = 0x4853454D  # arbitrary; the parser only skips it

#: Value the parser treats as "no parent" for 8-bit parent indices.
NO_PARENT_8 = 255


def row_vector_matrix(
    translation=(0.0, 0.0, 0.0),
    rotation_degrees: float = 0.0,
    axis: str = "z",
    scale=(1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Build a bone matrix in the layout NeoX files use.

    The contract matrix is assembled as a normal column-vector transform and
    then transposed once, which is what "translation sits in the last row"
    means in storage terms.

    Parameters:
    - translation: bone origin in model space.
    - rotation_degrees: rotation around ``axis``.
    - axis: one of ``"x"``, ``"y"``, ``"z"``.
    - scale: per-axis scale.

    Returns:
    - A 4x4 array laid out the way the file stores it.
    """
    angle = np.radians(rotation_degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    if axis == "x":
        basis = np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]], dtype=np.float64)
    elif axis == "y":
        basis = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]], dtype=np.float64)
    else:
        basis = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float64)

    column_vector = np.identity(4)
    column_vector[:3, :3] = basis * np.asarray(scale, dtype=np.float64)
    column_vector[:3, 3] = np.asarray(translation, dtype=np.float64)
    return column_vector.T.copy()


def contract_from_storage(stored: np.ndarray) -> np.ndarray:
    """Independent re-implementation of the row-vector reinterpretation."""
    return np.asarray(stored, dtype=np.float64).T.copy()


@dataclass
class SyntheticMesh:
    """A synthetic mesh plus the values the parser is expected to recover."""

    data: bytes
    vertex_count: int
    face_count: int
    positions: list[tuple[float, float, float]]
    normals: list[tuple[float, float, float]]
    faces: list[tuple[int, int, int]]
    uvs: list[tuple[float, float]]
    joints: list[tuple[int, int, int, int]]
    weights: list[tuple[float, float, float, float]]
    bone_parents: list[int]
    bone_names: list[str]
    bone_matrices: list[np.ndarray]
    joint_index_bits: int
    expected_type: int


def build_mesh_file(
    *,
    positions,
    normals,
    faces,
    uvs,
    joints,
    weights,
    bone_parents,
    bone_names,
    bone_matrices,
    joint_index_bits: int = 8,
    version: int = 5,
    has_bones: int = 4,
) -> SyntheticMesh:
    """
    Serialise a ``.mesh`` blob the real parser can read.

    The sizes are arranged so ``identify_mesh_type`` classifies the file as
    type 4 (8-bit joint indices) or type 5 (16-bit), which is the pair the
    joint-width regression needs.

    Parameters:
    - positions / normals: per-vertex 3-tuples, stored as half floats.
    - faces: triangles as vertex index 3-tuples.
    - uvs: one UV pair per vertex.
    - joints / weights: four influences per vertex.
    - bone_parents: parent index per bone, ``-1`` for a root.
    - bone_names: bone names, at most 31 characters.
    - bone_matrices: 4x4 arrays in file layout, see :func:`row_vector_matrix`.
    - joint_index_bits: 8 or 16.
    - version / has_bones: header fields.

    Returns:
    - A :class:`SyntheticMesh` carrying the bytes and the expected values.
    """
    if joint_index_bits not in (8, 16):
        raise ValueError("joint_index_bits must be 8 or 16")

    vertex_count = len(positions)
    face_count = len(faces)
    bone_count = len(bone_parents)
    uv_layers = 1

    header = bytearray()
    header += struct.pack("<I", MAGIC)
    header += struct.pack("<HHHH", version, 0x0500, has_bones, 0x0000)

    if has_bones in (1, 4):
        header += struct.pack("<H", bone_count)
        for parent in bone_parents:
            header += struct.pack("<B", NO_PARENT_8 if parent == -1 else parent)
        for name in bone_names:
            encoded = name.encode("ascii")
            if len(encoded) > 31:
                raise ValueError(f"bone name {name!r} does not fit in 32 bytes")
            header += encoded + b"\x00" * (32 - len(encoded))
        header += struct.pack("<B", 0)  # bone_extra_info
        for matrix in bone_matrices:
            header += struct.pack(
                "<16f", *np.asarray(matrix, dtype=np.float64).reshape(-1)
            )
        header += struct.pack("<B", 0)  # flag1, must be zero

    # The ending address field is patched once the mesh block length is known.
    ending_address_offset = len(header)
    header += struct.pack("<I", 0)
    main_data_offset = len(header)

    body = bytearray()
    # Two count headers: the parser reads the first pair, sees must_be_one == 1
    # and then re-reads the authoritative pair.
    body += struct.pack("<II", vertex_count, face_count)
    body += struct.pack("<BBH", uv_layers, 0, 1)
    body += struct.pack("<II", vertex_count, face_count)

    for position in positions:
        body += struct.pack("<3e", *position)
    for normal in normals:
        body += struct.pack("<3e", *normal)

    body += struct.pack("<H", 0)  # _flag == 0: no extra per-vertex block

    for face in faces:
        body += struct.pack("<3H", *face)

    for uv in uvs:
        body += struct.pack("<2f", *uv)

    joint_format = "<4H" if joint_index_bits == 16 else "<4B"
    for joint in joints:
        body += struct.pack(joint_format, *joint)
    for weight in weights:
        body += struct.pack("<4f", *weight)

    ending_address = main_data_offset + len(body)
    header[ending_address_offset : ending_address_offset + 4] = struct.pack(
        "<I", ending_address
    )

    table = bytearray()
    table += struct.pack("<H", 1)  # one mesh inside
    table += struct.pack("<I", main_data_offset)

    expected_parents = [-1 if parent == NO_PARENT_8 else parent for parent in bone_parents]

    return SyntheticMesh(
        data=bytes(header + body + table),
        vertex_count=vertex_count,
        face_count=face_count,
        positions=[tuple(np.float16(value).item() for value in p) for p in positions],
        normals=[tuple(np.float16(value).item() for value in n) for n in normals],
        faces=[tuple(face) for face in faces],
        uvs=[tuple(uv) for uv in uvs],
        joints=[tuple(joint) for joint in joints],
        weights=[tuple(np.float32(w).item() for w in weight) for weight in weights],
        bone_parents=expected_parents,
        bone_names=list(bone_names),
        bone_matrices=[np.asarray(m, dtype=np.float64) for m in bone_matrices],
        joint_index_bits=joint_index_bits,
        expected_type=5 if joint_index_bits == 16 else 4,
    )


def make_mesh_data(
    *,
    positions,
    faces,
    bone_parents=None,
    bone_names=None,
    bone_matrices=None,
    joints=None,
    weights=None,
    normals=None,
    uvs=None,
    joint_index_bits: int = 8,
    has_bones: int = 4,
    version: int = 5,
    mesh_type: int = 4,
) -> MeshData:
    """
    Build a :class:`MeshData` straight from Python values.

    Parameters mirror the fields of the dataclass; missing normals default to
    ``(0, 0, 1)`` and missing UVs to ``(0, 0)``.

    Returns:
    - A populated :class:`MeshData`.
    """
    vertex_count = len(positions)
    normals = normals if normals is not None else [(0.0, 0.0, 1.0)] * vertex_count
    uvs = uvs if uvs is not None else [(0.0, 0.0)] * vertex_count

    rigged = bone_parents is not None
    bones = Bones(
        has_bones=has_bones if rigged else 0,
        parents=list(bone_parents) if rigged else [],
        names=list(bone_names) if rigged else [],
        matrix=[np.asarray(m, dtype=np.float64) for m in bone_matrices]
        if rigged
        else [],
        count=len(bone_parents) if rigged else 0,
        joint_index_bits=joint_index_bits,
        joints=[tuple(joint) for joint in joints] if rigged else [],
        weights=[tuple(weight) for weight in weights] if rigged else [],
    )

    return MeshData(
        version=version,
        type=mesh_type,
        mesh=Mesh(
            vertexes=vertex_count,
            faces=len(faces),
            position=[tuple(p) for p in positions],
            normal=[tuple(n) for n in normals],
            face=[tuple(f) for f in faces],
            uv=[tuple(uv) for uv in uvs],
        ),
        bones=bones,
    )


def asymmetric_character(joint_index_bits: int = 8) -> MeshData:
    """
    A small asymmetric rig: root, two arms, each arm with a forearm child.

    Every limb has a distinct length so a left/right swap cannot pass
    unnoticed, and each vertex is bound to exactly one bone so a moved bone has
    an exactly predictable effect.

    Bone layout (positions are bone origins in model space)::

        0 root       (0, 0, 0)
        1 arm_l      (1, 2, 0)        2 arm_r      (-3, 2, 0)
        3 forearm_l  (1, 4, 0)        4 forearm_r  (-3, 5, 0)

    Note the parents deliberately appear *after* some of their children in the
    arrays, which is the ordering that used to drop branches from the scene.

    Returns:
    - A rigged :class:`MeshData`.
    """
    # Children before parents on purpose.
    names = ["forearm_l", "arm_l", "root", "forearm_r", "arm_r"]
    origins = {
        "root": (0.0, 0.0, 0.0),
        "arm_l": (1.0, 2.0, 0.0),
        "arm_r": (-3.0, 2.0, 0.0),
        "forearm_l": (1.0, 4.0, 0.0),
        "forearm_r": (-3.0, 5.0, 0.0),
    }
    index_of = {name: index for index, name in enumerate(names)}
    parent_of = {
        "root": -1,
        "arm_l": "root",
        "arm_r": "root",
        "forearm_l": "arm_l",
        "forearm_r": "arm_r",
    }
    parents = [
        -1 if parent_of[name] == -1 else index_of[parent_of[name]] for name in names
    ]
    matrices = [row_vector_matrix(origins[name]) for name in names]

    # One vertex per bone, offset from its bone origin so a rotation around the
    # bone pivot produces a different result than one around the model origin.
    positions = []
    joints = []
    weights = []
    for name in names:
        origin = origins[name]
        positions.append((origin[0] + 0.5, origin[1] + 0.25, origin[2] + 0.125))
        joints.append((index_of[name], 0, 0, 0))
        weights.append((1.0, 0.0, 0.0, 0.0))

    faces = [(0, 1, 2), (2, 3, 4), (0, 2, 4)]

    return make_mesh_data(
        positions=positions,
        faces=faces,
        bone_parents=parents,
        bone_names=names,
        bone_matrices=matrices,
        joints=joints,
        weights=weights,
        joint_index_bits=joint_index_bits,
        mesh_type=5 if joint_index_bits == 16 else 4,
    )


def multi_root_mesh():
    """Two roots, one with a child: nothing may hang off the wrong tree."""
    return make_mesh_data(
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (5.5, 0.5, 0.0)],
        faces=[(0, 1, 2), (1, 2, 3)],
        bone_parents=[-1, -1, 0],
        bone_names=["root_a", "root_b", "child_a"],
        bone_matrices=[
            row_vector_matrix((0.0, 0.0, 0.0)),
            row_vector_matrix((5.0, 0.0, 0.0)),
            row_vector_matrix((0.0, 3.0, 0.0), 25.0),
        ],
        joints=[(0, 0, 0, 0), (2, 0, 0, 0), (0, 0, 0, 0), (1, 0, 0, 0)],
        weights=[(1.0, 0.0, 0.0, 0.0)] * 4,
    )


def reversed_storage_mesh():
    """Parents stored strictly after their children."""
    return make_mesh_data(
        positions=[(0.5, 4.0, 0.0), (0.5, 2.0, 0.0), (0.5, 0.0, 0.0)],
        faces=[(0, 1, 2)],
        bone_parents=[1, 2, -1],
        bone_names=["tip", "mid", "base"],
        bone_matrices=[
            row_vector_matrix((0.0, 4.0, 0.0), -30.0),
            row_vector_matrix((0.0, 2.0, 0.0), 15.0),
            row_vector_matrix((0.0, 0.0, 0.0)),
        ],
        joints=[(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0)],
        weights=[(1.0, 0.0, 0.0, 0.0)] * 3,
    )


def bone_255_mesh():
    """257 bones with a real influence on bone 255, which 8-bit data reserves."""
    bone_count = 257
    return make_mesh_data(
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
        bone_parents=[-1] + list(range(bone_count - 1)),
        bone_names=[f"bone_{index}" for index in range(bone_count)],
        bone_matrices=[
            row_vector_matrix((index * 0.05, index * 0.02, 0.0))
            for index in range(bone_count)
        ],
        joints=[
            (255, 65535, 65535, 65535),
            (256, 65535, 65535, 65535),
            (0, 65535, 65535, 65535),
        ],
        weights=[(1.0, 0.0, 0.0, 0.0)] * 3,
        joint_index_bits=16,
        mesh_type=5,
    )


def _box(low, high):
    """
    A box as 24 vertices with flat outward normals.

    Each face is wound counter-clockwise seen from outside, which is checked
    against its normal rather than trusted, so the box renders solid with
    back faces culled.
    """
    (x0, y0, z0), (x1, y1, z1) = low, high
    sides = [
        ((1.0, 0.0, 0.0), [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]),
        ((-1.0, 0.0, 0.0), [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)]),
        ((0.0, 1.0, 0.0), [(x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)]),
        ((0.0, -1.0, 0.0), [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]),
        ((0.0, 0.0, 1.0), [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]),
        ((0.0, 0.0, -1.0), [(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)]),
    ]
    positions, normals, quads = [], [], []
    for normal, corners in sides:
        a, b, c = (np.asarray(corner) for corner in corners[:3])
        if np.dot(np.cross(b - a, c - a), normal) < 0.0:
            corners = corners[::-1]
        start = len(positions)
        positions.extend(corners)
        normals.extend([normal] * 4)
        quads.append((start, start + 1, start + 2, start + 3))
    return positions, normals, quads


def box_figure() -> MeshData:
    """
    A blocky figure that looks the same to no mirror image of itself.

    Meant for looking at, not only for asserting on: it has a nose on the
    face, toes on the feet, and a staff in the **left** hand only. It stands on
    Y = 0 with Y up and faces +Z, which is the glTF front, so its left hand is
    at +X. Every box is bound entirely to one bone, and the bones carry L/R
    names the way real rigs do.

    Bone origins::

        root (0, 0, 0) - pelvis (0, 1, 0) - spine (0, 1.5, 0) - head (0, 2.25, 0)
        upperarm_l (0.35, 2.1, 0) - forearm_l (0.9, 2.1, 0)   the staff hand
        upperarm_r (-0.35, 2.1, 0) - forearm_r (-0.9, 2.1, 0)
        thigh_l (0.18, 1, 0) - calf_l (0.18, 0.5, 0), and the same at -X

    Returns:
    - A rigged :class:`MeshData`.
    """
    bones = [
        ("root", None, (0.0, 0.0, 0.0)),
        ("pelvis", "root", (0.0, 1.0, 0.0)),
        ("spine", "pelvis", (0.0, 1.5, 0.0)),
        ("head", "spine", (0.0, 2.25, 0.0)),
        ("upperarm_l", "spine", (0.35, 2.1, 0.0)),
        ("forearm_l", "upperarm_l", (0.9, 2.1, 0.0)),
        ("upperarm_r", "spine", (-0.35, 2.1, 0.0)),
        ("forearm_r", "upperarm_r", (-0.9, 2.1, 0.0)),
        ("thigh_l", "pelvis", (0.18, 1.0, 0.0)),
        ("calf_l", "thigh_l", (0.18, 0.5, 0.0)),
        ("thigh_r", "pelvis", (-0.18, 1.0, 0.0)),
        ("calf_r", "thigh_r", (-0.18, 0.5, 0.0)),
    ]
    index_of = {name: index for index, (name, _, _) in enumerate(bones)}

    def limbs(side, sign):
        """Arm and leg boxes for one side; sign is +1 for left, -1 for right."""

        def span(a, b):
            return (min(sign * a, sign * b), max(sign * a, sign * b))

        def part(x_range, y_range, z_range, bone):
            (xa, xb), (ya, yb), (za, zb) = span(*x_range), y_range, z_range
            return ((xa, ya, za), (xb, yb, zb), f"{bone}_{side}")

        return [
            part((0.35, 0.9), (2.0, 2.2), (-0.08, 0.08), "upperarm"),
            part((0.9, 1.4), (2.02, 2.18), (-0.07, 0.07), "forearm"),
            part((0.06, 0.3), (0.5, 0.95), (-0.1, 0.1), "thigh"),
            part((0.07, 0.29), (0.08, 0.5), (-0.09, 0.09), "calf"),
            part((0.07, 0.29), (0.0, 0.08), (-0.1, 0.3), "calf"),  # foot, toes to +Z
        ]

    parts = [
        ((-0.33, 0.9, -0.15), (0.33, 1.2, 0.15), "pelvis"),
        ((-0.35, 1.2, -0.17), (0.35, 2.2, 0.17), "spine"),
        ((-0.2, 2.25, -0.2), (0.2, 2.7, 0.2), "head"),
        ((-0.05, 2.38, 0.2), (0.05, 2.5, 0.34), "head"),  # the nose: the front
        ((1.4, 1.2, -0.04), (1.5, 3.1, 0.04), "forearm_l"),  # the staff: the left
        *limbs("l", 1.0),
        *limbs("r", -1.0),
    ]

    positions, normals, faces, joints = [], [], [], []
    for low, high, bone in parts:
        box_positions, box_normals, quads = _box(low, high)
        start = len(positions)
        positions.extend(box_positions)
        normals.extend(box_normals)
        joints.extend([(index_of[bone], 0, 0, 0)] * len(box_positions))
        for a, b, c, d in quads:
            faces.append((start + a, start + b, start + c))
            faces.append((start + a, start + c, start + d))

    return make_mesh_data(
        positions=positions,
        normals=normals,
        faces=faces,
        bone_parents=[-1 if parent is None else index_of[parent] for _, parent, _ in bones],
        bone_names=[name for name, _, _ in bones],
        bone_matrices=[row_vector_matrix(origin) for _, _, origin in bones],
        joints=joints,
        weights=[(1.0, 0.0, 0.0, 0.0)] * len(positions),
    )


def build_rgis_file(
    *,
    reference,
    clips,
    version: int = 2,
    unknown: int = 774,
    half_rotation: bool = False,
) -> bytes:
    """
    Serialise an RGIS animation container the real reader can parse.

    Parameters:
    - reference: list of ``(name, translation, rotation_xyzw, scale)``.
    - half_rotation: store rotations as float16, which the ``0x0002`` layout
      bit selects.
    - clips: list of dicts with ``name``, ``fps``, ``times`` (seconds),
      ``tracks`` and optionally ``layout_flag`` and ``root``. Each track is
      ``(bone_name, translation, rotation, scale)`` where a channel is either a
      single value (constant) or one value per key (animated).

    Returns:
    - The bytes of a ``.gis`` file.
    """

    def name_field(text: str) -> bytes:
        encoded = text.encode("ascii")
        if len(encoded) > 31:
            raise ValueError(f"name {text!r} does not fit in 32 bytes")
        return encoded + b"\x00" * (32 - len(encoded))

    out = bytearray(b"RGIS")
    out += struct.pack("<HHHI", version, unknown, len(clips), len(reference))
    for entry in reference:
        out += name_field(entry[0])
    for _, translation, rotation, scale in reference:
        out += struct.pack("<10f", *translation, *rotation, *scale)
    out += struct.pack("<I", 0)

    for clip in clips:
        tracks = clip["tracks"]
        times = np.asarray(clip["times"], dtype=np.float64)
        out += name_field(clip["name"])
        out += name_field("")
        out += name_field(clip.get("root", "root"))
        out += struct.pack("<H", len(tracks))
        for track in tracks:
            out += name_field(track[0])
        out += struct.pack(
            "<8H",
            clip.get("fps", 30),
            0,
            0,
            7,
            0,
            0,
            clip.get("layout_flag", 4),
            len(times),
        )
        # Times go out in milliseconds.
        out += struct.pack(f"<{len(times)}f", *(times * 1000.0))

        for _, translation, rotation, scale in tracks:
            translation = np.atleast_2d(np.asarray(translation, dtype=np.float64))
            rotation = np.atleast_2d(np.asarray(rotation, dtype=np.float64))
            scale = np.atleast_2d(np.asarray(scale, dtype=np.float64))
            out += struct.pack(
                "<4B",
                1 if len(translation) > 1 else 0,
                1 if len(rotation) > 1 else 0,
                1 if len(scale) > 1 else 0,
                0,
            )
            out += translation.astype("<f4").tobytes()
            out += rotation.astype("<f2" if half_rotation else "<f4").tobytes()
            out += scale.astype("<f2").tobytes()
        out += struct.pack("<B", 0)

    return bytes(out)
