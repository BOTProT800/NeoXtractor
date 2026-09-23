"""
Regression tests for the binary mesh reader.

Each test here reproduces a defect that was present in the reader, with the
expected values worked out from the byte layout rather than from the reader
itself.
"""

import numpy as np
import pytest

from core.mesh_loader.parsers.new_parser import MeshParser0
from tests.support.synthetic import (
    build_mesh_file,
    make_mesh_data,
    row_vector_matrix,
)


def _two_bone_payload(joint_index_bits, joints, weights):
    """A two-vertex, one-triangle mesh with two bones."""
    return build_mesh_file(
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        normals=[(0.0, 0.0, 1.0)] * 3,
        faces=[(0, 1, 2)],
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        joints=joints,
        weights=weights,
        bone_parents=[-1, 0],
        bone_names=["root", "child"],
        bone_matrices=[
            row_vector_matrix((0.0, 0.0, 0.0)),
            row_vector_matrix((2.0, 0.0, 0.0)),
        ],
        joint_index_bits=joint_index_bits,
    )


@pytest.mark.parametrize("joint_index_bits", [8, 16])
def test_joint_indices_and_weights_survive_the_round_trip(joint_index_bits):
    """
    Type 5 reads its joint indices as uint16.

    Before the fix the type dispatch was a single elif chain: type 5 matched
    the half-precision branch and never reached the one that widens the joint
    index, so it consumed 4 bytes where the file holds 8 and started decoding
    weights four bytes early. The first vertex came back with joints
    ``[0, 0, 1, 0]`` and weights ``[0, 9.18e-41, 0, 0.75]``.
    """
    joints = [(0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0)]
    weights = [(0.75, 0.25, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.0, 0.0)]
    fixture = _two_bone_payload(joint_index_bits, joints, weights)

    parsed = MeshParser0().parse(fixture.data)

    assert parsed.type == fixture.expected_type
    assert parsed.bones.joint_index_bits == joint_index_bits
    assert [tuple(j) for j in parsed.bones.joints] == joints
    assert [tuple(w) for w in parsed.bones.weights] == pytest.approx(weights)


def test_positions_normals_and_faces_are_read_where_they_live():
    """A misread joint block would also shift everything after it."""
    fixture = _two_bone_payload(
        16,
        [(0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0)],
        [(0.75, 0.25, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.0, 0.0)],
    )

    parsed = MeshParser0().parse(fixture.data)

    assert parsed.mesh.vertexes == fixture.vertex_count
    assert parsed.mesh.faces == fixture.face_count
    assert [tuple(p) for p in parsed.mesh.position] == pytest.approx(fixture.positions)
    assert [tuple(n) for n in parsed.mesh.normal] == pytest.approx(fixture.normals)
    assert [tuple(f) for f in parsed.mesh.face] == fixture.faces
    assert [tuple(uv) for uv in parsed.mesh.uv] == pytest.approx(fixture.uvs)


def test_bone_index_255_is_a_real_bone_with_16_bit_indices():
    """
    With 16-bit joint indices only 65535 marks an empty slot.

    The old exporter treated both 255 and 65535 as sentinels, so a legitimate
    influence on bone 255 was rewritten to the root node index.
    """
    bone_count = 257
    parents = [-1] + list(range(bone_count - 1))
    names = [f"bone_{index}" for index in range(bone_count)]
    matrices = [row_vector_matrix((float(index), 0.0, 0.0)) for index in range(bone_count)]

    fixture = build_mesh_file(
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        normals=[(0.0, 0.0, 1.0)] * 3,
        faces=[(0, 1, 2)],
        uvs=[(0.0, 0.0)] * 3,
        joints=[(255, 65535, 65535, 65535), (0, 0, 0, 0), (256, 0, 0, 0)],
        weights=[(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
        bone_parents=parents,
        bone_names=names,
        bone_matrices=matrices,
        joint_index_bits=16,
    )

    parsed = MeshParser0().parse(fixture.data)

    assert parsed.bones.joint_index_bits == 16
    assert parsed.bones.joint_index_sentinel == 65535
    assert tuple(parsed.bones.joints[0]) == (255, 65535, 65535, 65535)
    assert tuple(parsed.bones.joints[2]) == (256, 0, 0, 0)


def test_multiple_roots_keep_the_bone_count_in_sync():
    """
    The synthetic root has to be counted.

    The parser appends ``dummy_root`` when it finds more than one root, growing
    parents, names and matrices. ``count`` stayed behind, and the exporter used
    it to size the inverse bind accessor.
    """
    fixture = build_mesh_file(
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        normals=[(0.0, 0.0, 1.0)] * 3,
        faces=[(0, 1, 2)],
        uvs=[(0.0, 0.0)] * 3,
        joints=[(0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0)],
        weights=[(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
        bone_parents=[-1, -1],
        bone_names=["root_a", "root_b"],
        bone_matrices=[
            row_vector_matrix((0.0, 0.0, 0.0)),
            row_vector_matrix((5.0, 0.0, 0.0)),
        ],
        joint_index_bits=8,
    )

    parsed = MeshParser0().parse(fixture.data)

    assert parsed.bones.names == ["root_a", "root_b", "dummy_root"]
    assert parsed.bones.parents == [2, 2, -1]
    assert parsed.bones.count == 3
    assert len(parsed.bones.matrix) == 3
    assert parsed.validate_rig() == []


@pytest.mark.parametrize("joint_index_bits", [8, 16])
def test_the_reader_consumes_exactly_the_influence_block(joint_index_bits):
    """
    The stream has to end where the mesh data block ends.

    This is the direct guard against a misaligned read: the type-5 defect
    consumed four bytes too few for the joints and four too many for the
    weights, so the final offset was wrong even though every array had the
    right length.
    """
    import io

    joints = [(0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0)]
    weights = [(0.75, 0.25, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.0, 0.0)]
    fixture = _two_bone_payload(joint_index_bits, joints, weights)

    stream = io.BytesIO(fixture.data)
    MeshParser0()._parse_mesh_testing(stream)

    # The mesh index table sits right after the mesh data, and the parser must
    # stop exactly at its start.
    table_size = 2 + 4  # uint16 count + one uint32 offset
    assert stream.tell() == len(fixture.data) - table_size


def test_bone_matrices_are_stored_verbatim():
    """The reader must not reinterpret matrices; that belongs to the exporter."""
    stored = row_vector_matrix((2.0, 3.0, 4.0), rotation_degrees=30.0)
    fixture = build_mesh_file(
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        normals=[(0.0, 0.0, 1.0)] * 3,
        faces=[(0, 1, 2)],
        uvs=[(0.0, 0.0)] * 3,
        joints=[(0, 0, 0, 0)] * 3,
        weights=[(1.0, 0.0, 0.0, 0.0)] * 3,
        bone_parents=[-1],
        bone_names=["root"],
        bone_matrices=[stored],
        joint_index_bits=8,
    )

    parsed = MeshParser0().parse(fixture.data)

    assert np.allclose(parsed.bones.matrix[0], stored, atol=1e-6)
    # The translation sits in the last row, which is what "row-vector storage"
    # means at the byte level.
    assert np.allclose(parsed.bones.matrix[0][3, :3], (2.0, 3.0, 4.0), atol=1e-6)
    assert np.allclose(parsed.bones.matrix[0][:3, 3], (0.0, 0.0, 0.0), atol=1e-6)


def test_a_valid_triangle_passes_geometry_validation():
    """
    ``validate`` compared vertex indices against the *face* count.

    A three-vertex, one-triangle mesh -- perfectly valid -- was rejected
    because index 2 is not smaller than 1.
    """
    mesh = make_mesh_data(
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
    )

    assert mesh.validate_geometry() == []
    assert mesh.validate() is True


def test_geometry_validation_rejects_an_out_of_range_corner():
    mesh = make_mesh_data(
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 3)],
    )

    problems = mesh.validate_geometry()

    assert problems and "references vertex 3" in problems[0]


def test_rig_validation_reports_cycles_and_bad_matrices():
    mesh = make_mesh_data(
        positions=[(0.0, 0.0, 0.0)],
        faces=[(0, 0, 0)],
        bone_parents=[1, 0],
        bone_names=["a", "b"],
        bone_matrices=[np.identity(4), np.zeros((4, 4))],
        joints=[(0, 0, 0, 0)],
        weights=[(1.0, 0.0, 0.0, 0.0)],
    )

    problems = mesh.validate_rig()

    assert any("no root bone" in problem for problem in problems)
    assert any("singular" in problem for problem in problems)


def test_rig_validation_reports_a_stale_bone_count():
    mesh = make_mesh_data(
        positions=[(0.0, 0.0, 0.0)],
        faces=[(0, 0, 0)],
        bone_parents=[-1, 0],
        bone_names=["a", "b"],
        bone_matrices=[np.identity(4), np.identity(4)],
        joints=[(0, 0, 0, 0)],
        weights=[(1.0, 0.0, 0.0, 0.0)],
    )
    mesh.bones.count = 1

    problems = mesh.validate_rig()

    assert any("bone count field is 1" in problem for problem in problems)
