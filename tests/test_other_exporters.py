"""
Regression tests for the ASCII, PMX and SMD exporters.

All three read the bone translation out of the wrong slot of the stored
matrix, so every bone landed on the model origin, and all three walked the
hierarchy from a single root. The expected values here are computed from the
fixture's own numbers, not from the exporters.
"""

import numpy as np
import pytest

from core.mesh_converter.formats import ascii as mesh_ascii
from core.mesh_converter.formats import pmx as pmx_format
from core.mesh_converter.formats import smd as smd_format
from core.mesh_converter.skeleton import euler_xyz_from_matrix
from tests.support.synthetic import (
    asymmetric_character,
    make_mesh_data,
    row_vector_matrix,
)


def rotation(axis, degrees):
    angle = np.radians(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]], dtype=np.float64)
    if axis == "y":
        return np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]], dtype=np.float64)
    return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float64)


def recompose_euler(angles):
    """Rebuild a rotation from XYZ Euler angles: ``Rz @ Ry @ Rx``."""
    rx, ry, rz = angles
    return (
        rotation("z", np.degrees(rz))
        @ rotation("y", np.degrees(ry))
        @ rotation("x", np.degrees(rx))
    )


def rotated_rig():
    """A three-bone chain where every bone carries a different rotation."""
    return make_mesh_data(
        positions=[(0.5, 0.0, 0.0), (0.5, 2.0, 0.0), (0.5, 4.0, 0.0)],
        faces=[(0, 1, 2)],
        bone_parents=[-1, 0, 1],
        bone_names=["base", "mid", "tip"],
        bone_matrices=[
            row_vector_matrix((0.0, 0.0, 0.0), 20.0, axis="z"),
            row_vector_matrix((1.0, 2.0, 0.0), -35.0, axis="x"),
            row_vector_matrix((2.0, 4.0, 1.0), 50.0, axis="y"),
        ],
        joints=[(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0)],
        weights=[(1.0, 0.0, 0.0, 0.0)] * 3,
    )


def multi_root_rig():
    return make_mesh_data(
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
        # Child stored before both roots.
        bone_parents=[1, -1, -1],
        bone_names=["child_of_a", "root_a", "root_b"],
        bone_matrices=[
            row_vector_matrix((0.0, 3.0, 0.0)),
            row_vector_matrix((0.0, 0.0, 0.0)),
            row_vector_matrix((7.0, 0.0, 0.0)),
        ],
        joints=[(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0)],
        weights=[(1.0, 0.0, 0.0, 0.0)] * 3,
    )


def expected_global_origins(mesh):
    """Bone origins straight from the file: row-vector layout, last row."""
    return {
        name: np.asarray(matrix, dtype=np.float64)[3, :3]
        for name, matrix in zip(mesh.bones.names, mesh.bones.matrix)
    }


def parse_ascii_bones(payload):
    """Read back the bone block: name -> (position, quaternion)."""
    lines = payload.decode("utf-8").splitlines()
    count = int(lines[0])
    bones = {}
    cursor = 1
    for _ in range(count):
        name = lines[cursor]
        parent = int(lines[cursor + 1])
        values = [float(v) for v in lines[cursor + 2].split()]
        bones[name] = (np.array(values[:3]), np.array(values[3:7]), parent)
        cursor += 3
    return bones


def parse_smd(payload):
    """Read back the nodes and skeleton blocks of an SMD."""
    lines = payload.decode("utf-8").splitlines()
    nodes, skeleton, triangles = {}, {}, []
    section = None
    for line in lines:
        stripped = line.strip()
        if stripped in ("nodes", "skeleton", "triangles"):
            section = stripped
            continue
        if stripped == "end":
            section = None
            continue
        if section == "nodes":
            index, rest = stripped.split(" ", 1)
            name, parent = rest.rsplit(" ", 1)
            nodes[int(index)] = (name.strip('"'), int(parent))
        elif section == "skeleton" and not stripped.startswith("time"):
            parts = stripped.split()
            skeleton[int(parts[0])] = (
                np.array([float(v) for v in parts[1:4]]),
                np.array([float(v) for v in parts[4:7]]),
            )
        elif section == "triangles":
            parts = stripped.split()
            if len(parts) >= 9:
                triangles.append(parts)
    return nodes, skeleton, triangles


class TestAscii:
    def test_bone_positions_are_the_real_rest_origins(self):
        """
        The old code wrote ``matrix.flatten()[:3]``.

        That is the first row of the stored array, which is a basis vector,
        not a translation under any convention. For the asymmetric rig it
        produced ``1 0 0`` for every bone.
        """
        mesh = asymmetric_character()
        bones = parse_ascii_bones(mesh_ascii.convert(mesh))
        expected = expected_global_origins(mesh)

        for name, origin in expected.items():
            assert bones[name][0] == pytest.approx(origin, abs=1e-5), name

    def test_bones_are_not_all_at_the_origin(self):
        mesh = asymmetric_character()
        bones = parse_ascii_bones(mesh_ascii.convert(mesh))

        positions = np.array([value[0] for value in bones.values()])
        assert np.abs(positions).max() > 0.5

    def test_the_quaternion_is_the_real_orientation(self):
        """It used to be a hardcoded ``0 0 0 1`` for every bone."""
        mesh = rotated_rig()
        bones = parse_ascii_bones(mesh_ascii.convert(mesh))

        # Bone "base" is turned 20 degrees about Z: (0, 0, sin10, cos10).
        x, y, z, w = bones["base"][1]
        assert (x, y, z) == pytest.approx(
            (0.0, 0.0, np.sin(np.radians(10.0))), abs=1e-5
        )
        assert w == pytest.approx(np.cos(np.radians(10.0)), abs=1e-5)
        assert np.linalg.norm(bones["base"][1]) == pytest.approx(1.0, abs=1e-5)

    def test_parents_are_preserved(self):
        mesh = asymmetric_character()
        bones = parse_ascii_bones(mesh_ascii.convert(mesh))

        for index, name in enumerate(mesh.bones.names):
            assert bones[name][2] == mesh.bones.parents[index]

    def test_a_mesh_without_bones_still_exports(self):
        mesh = make_mesh_data(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[(0, 1, 2)],
        )

        assert mesh_ascii.convert(mesh).decode("utf-8").startswith("0\n")


class TestSmd:
    def test_the_skeleton_block_is_parent_relative(self):
        """
        SMD stores each bone relative to its parent.

        The old code wrote a global slot that was always zero, and treated it
        as if it were already parent-relative.
        """
        mesh = asymmetric_character()
        nodes, skeleton, _ = parse_smd(smd_format.convert(mesh))
        expected = expected_global_origins(mesh)
        index_of = {name: index for index, name in enumerate(mesh.bones.names)}

        for name, origin in expected.items():
            index = index_of[name]
            parent = mesh.bones.parents[index]
            reference = (
                np.zeros(3) if parent == -1 else expected[mesh.bones.names[parent]]
            )
            assert skeleton[index][0] == pytest.approx(origin - reference, abs=1e-5), name

    def test_euler_angles_reproduce_the_local_rotation(self):
        mesh = rotated_rig()
        _, skeleton, _ = parse_smd(smd_format.convert(mesh))

        # "base" is a root, so its local rotation is its global one: 20 deg Z.
        assert np.allclose(
            recompose_euler(skeleton[0][1]), rotation("z", 20.0), atol=1e-5
        )
        # "mid" sits under it, so the residual is the parent's turn undone.
        expected_local = rotation("z", 20.0).T @ rotation("x", -35.0)
        assert np.allclose(recompose_euler(skeleton[1][1]), expected_local, atol=1e-5)

    def test_rotations_are_not_hardcoded_to_zero(self):
        mesh = rotated_rig()
        _, skeleton, _ = parse_smd(smd_format.convert(mesh))

        assert max(np.abs(value[1]).max() for value in skeleton.values()) > 0.1

    def test_every_bone_is_declared_with_several_roots(self):
        """
        ``parents.index(-1)`` walked one root only.

        Bones under the other roots never reached the nodes block.
        """
        mesh = multi_root_rig()
        nodes, skeleton, _ = parse_smd(smd_format.convert(mesh))

        assert len(nodes) == len(mesh.bones.names)
        assert {value[0] for value in nodes.values()} == set(mesh.bones.names)
        assert len(skeleton) == len(mesh.bones.names)
        for index, parent in enumerate(mesh.bones.parents):
            assert nodes[index][1] == parent

    def test_vertex_weight_links_are_written(self):
        """The old output kept only the dominant bone and dropped the weights."""
        mesh = make_mesh_data(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[(0, 1, 2)],
            bone_parents=[-1, 0],
            bone_names=["root", "child"],
            bone_matrices=[
                row_vector_matrix((0.0, 0.0, 0.0)),
                row_vector_matrix((0.0, 2.0, 0.0)),
            ],
            joints=[(0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0)],
            weights=[(0.75, 0.25, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
        )

        _, _, triangles = parse_smd(smd_format.convert(mesh))

        first = next(row for row in triangles if int(row[0]) == 0 and row[10:])
        link_count = int(first[9])
        links = {
            int(first[10 + 2 * i]): float(first[11 + 2 * i]) for i in range(link_count)
        }
        assert links == pytest.approx({0: 0.75, 1: 0.25})

    def test_a_sentinel_slot_does_not_become_a_bone(self):
        mesh = make_mesh_data(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[(0, 1, 2)],
            bone_parents=[-1, 0],
            bone_names=["root", "child"],
            bone_matrices=[
                row_vector_matrix((0.0, 0.0, 0.0)),
                row_vector_matrix((0.0, 2.0, 0.0)),
            ],
            joints=[(1, 255, 255, 255), (0, 255, 255, 255), (0, 255, 255, 255)],
            weights=[(1.0, 0.0, 0.0, 0.0)] * 3,
        )

        _, _, triangles = parse_smd(smd_format.convert(mesh))

        for row in triangles:
            link_count = int(row[9])
            for i in range(link_count):
                assert int(row[10 + 2 * i]) < len(mesh.bones.names)

    def test_a_mesh_without_bones_still_exports(self):
        mesh = make_mesh_data(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[(0, 1, 2)],
        )

        nodes, skeleton, triangles = parse_smd(smd_format.convert(mesh))

        assert nodes == {0: ("root", -1)}
        assert len(triangles) == 3


class TestPmx:
    def read_back(self, payload, tmp_path):
        """Parse a PMX with pymeshio, the same library that wrote it."""
        pymeshio_reader = pytest.importorskip("pymeshio.pmx.reader")
        target = tmp_path / "model.pmx"
        target.write_bytes(payload)
        return pymeshio_reader.read_from_file(str(target))

    def test_bone_positions_are_absolute_rest_origins(self, tmp_path):
        mesh = asymmetric_character()
        model = self.read_back(pmx_format.convert(mesh), tmp_path)
        expected = expected_global_origins(mesh)

        written = {
            bone.name: np.array([bone.position.x, bone.position.y, bone.position.z])
            for bone in model.bones
        }
        assert set(written) == set(expected)
        for name, origin in expected.items():
            assert written[name] == pytest.approx(origin, abs=1e-4), name

    def test_bones_are_not_all_at_the_origin(self, tmp_path):
        mesh = asymmetric_character()
        model = self.read_back(pmx_format.convert(mesh), tmp_path)

        positions = np.array(
            [[b.position.x, b.position.y, b.position.z] for b in model.bones]
        )
        assert np.abs(positions).max() > 0.5

    def test_parents_come_before_their_children(self, tmp_path):
        """PMX requires it, and several roots must not drop a branch."""
        mesh = multi_root_rig()
        model = self.read_back(pmx_format.convert(mesh), tmp_path)

        assert len(model.bones) == len(mesh.bones.names)
        for index, bone in enumerate(model.bones):
            assert bone.parent_index < index

    def test_the_hierarchy_matches_the_source(self, tmp_path):
        mesh = asymmetric_character()
        model = self.read_back(pmx_format.convert(mesh), tmp_path)

        position_of = {bone.name: index for index, bone in enumerate(model.bones)}
        for index, name in enumerate(mesh.bones.names):
            source_parent = mesh.bones.parents[index]
            written_parent = model.bones[position_of[name]].parent_index
            if source_parent == -1:
                assert written_parent == -1
            else:
                assert written_parent == position_of[mesh.bones.names[source_parent]]

    def test_vertex_weights_reference_the_remapped_bones(self, tmp_path):
        mesh = asymmetric_character()
        model = self.read_back(pmx_format.convert(mesh), tmp_path)

        position_of = {bone.name: index for index, bone in enumerate(model.bones)}
        for vertex_index, vertex in enumerate(model.vertices):
            source_bone = mesh.bones.joints[vertex_index][0]
            expected = position_of[mesh.bones.names[source_bone]]
            assert vertex.deform.index0 == expected

    def test_a_mesh_without_bones_still_exports(self, tmp_path):
        mesh = make_mesh_data(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[(0, 1, 2)],
        )
        model = self.read_back(pmx_format.convert(mesh), tmp_path)

        assert [bone.name for bone in model.bones] == ["root"]


class TestEulerHelper:
    @pytest.mark.parametrize(
        "axis, degrees", [("x", 30.0), ("y", -47.0), ("z", 115.0)]
    )
    def test_single_axis_round_trips(self, axis, degrees):
        source = np.identity(4)
        source[:3, :3] = rotation(axis, degrees)

        assert np.allclose(
            recompose_euler(euler_xyz_from_matrix(source)), source[:3, :3], atol=1e-9
        )

    def test_a_combined_rotation_round_trips(self):
        source = np.identity(4)
        source[:3, :3] = (
            rotation("z", 40.0) @ rotation("y", 25.0) @ rotation("x", -60.0)
        )

        assert np.allclose(
            recompose_euler(euler_xyz_from_matrix(source)), source[:3, :3], atol=1e-9
        )

    def test_gimbal_lock_still_reproduces_the_rotation(self):
        """At ry = 90 degrees the X and Z split is arbitrary but the result is not."""
        source = np.identity(4)
        source[:3, :3] = rotation("z", 10.0) @ rotation("y", 90.0)

        assert np.allclose(
            recompose_euler(euler_xyz_from_matrix(source)), source[:3, :3], atol=1e-9
        )

    def test_scale_does_not_leak_into_the_angles(self):
        source = np.identity(4)
        source[:3, :3] = rotation("y", 33.0) @ np.diag([2.0, 3.0, 0.5])

        assert np.allclose(
            recompose_euler(euler_xyz_from_matrix(source)), rotation("y", 33.0), atol=1e-9
        )
