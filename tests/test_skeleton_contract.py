"""
Tests for the skeleton contract.

The maths is checked against values computed by hand, not against the module's
own output. Translation-only fixtures are never used on their own: a wrong
multiplication order is invisible when every rotation is identity.
"""

import numpy as np
import pytest

from core.mesh_converter.skeleton import (
    IDENTITY_CONVERSION,
    NEOX_TO_GLTF,
    MatrixRole,
    MatrixStorage,
    SkeletonError,
    build_skeleton,
    compose_trs,
    decompose_trs,
    detect_matrix_storage,
    quaternion_from_matrix,
)
from tests.support.synthetic import contract_from_storage, row_vector_matrix


def rotation_z(degrees):
    angle = np.radians(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def column_vector_transform(translation, degrees=0.0):
    matrix = np.identity(4)
    matrix[:3, :3] = rotation_z(degrees)
    matrix[:3, 3] = translation
    return matrix


class TestStorageDetection:
    def test_row_vector_storage_is_detected_from_the_affine_column(self):
        matrices = [row_vector_matrix((1.0, 2.0, 3.0), 15.0)]

        storage, reason = detect_matrix_storage(matrices)

        assert storage is MatrixStorage.ROW_VECTOR
        assert "last column" in reason

    def test_column_vector_storage_is_detected_from_the_affine_row(self):
        matrices = [column_vector_transform((1.0, 2.0, 3.0), 15.0)]

        storage, reason = detect_matrix_storage(matrices)

        assert storage is MatrixStorage.COLUMN_VECTOR
        assert "last row" in reason

    def test_a_rotation_only_skeleton_is_reported_as_ambiguous(self):
        """Both layouts look identical without translation; say so."""
        matrices = [column_vector_transform((0.0, 0.0, 0.0), 40.0)]

        storage, reason = detect_matrix_storage(matrices)

        assert storage is MatrixStorage.ROW_VECTOR
        assert "indistinguishable" in reason


class TestTRSDecomposition:
    def test_translation_and_rotation_round_trip(self):
        original = column_vector_transform((2.0, 3.0, 4.0), 37.0)

        translation, rotation, scale, error = decompose_trs(original)

        assert error == pytest.approx(0.0, abs=1e-12)
        assert translation == pytest.approx([2.0, 3.0, 4.0])
        assert scale == pytest.approx([1.0, 1.0, 1.0])
        assert np.allclose(compose_trs(translation, rotation, scale), original)

    def test_non_uniform_scale_round_trips(self):
        original = column_vector_transform((1.0, 0.0, 0.0), 20.0)
        original[:3, :3] = original[:3, :3] @ np.diag([2.0, 3.0, 0.5])

        translation, rotation, scale, error = decompose_trs(original)

        assert error == pytest.approx(0.0, abs=1e-9)
        assert np.allclose(compose_trs(translation, rotation, scale), original)

    def test_a_mirrored_basis_keeps_its_reflection(self):
        original = np.identity(4)
        original[:3, :3] = np.diag([-1.0, 1.0, 1.0])

        translation, rotation, scale, error = decompose_trs(original)

        assert error == pytest.approx(0.0, abs=1e-12)
        assert scale[0] < 0.0
        assert np.allclose(compose_trs(translation, rotation, scale), original)

    def test_shear_is_reported_rather_than_silently_dropped(self):
        original = np.identity(4)
        original[0, 1] = 0.5

        _, _, _, error = decompose_trs(original)

        assert error > 1e-3

    def test_quaternion_matches_a_known_rotation(self):
        # A 90 degree turn around Z is (0, 0, sin45, cos45).
        quaternion = quaternion_from_matrix(rotation_z(90.0))

        assert quaternion == pytest.approx([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])


class TestHierarchy:
    def test_local_transform_is_parent_relative_with_rotation(self):
        """
        ``L = inverse(G_parent) @ G_child``, checked against hand maths.

        Parent at (1, 0, 0) turned 90 degrees about Z, child at (3, 0, 0) with
        the same rotation. The offset between them is 2 units along world +X.
        The parent's rotation maps its local X onto world +Y, so its local
        frame sees that offset as ``R_parent^T @ (2, 0, 0) = (0, -2, 0)``, with
        no residual rotation because both bones are turned the same way.

        Getting the multiplication order backwards yields ``(0, +2, 0)`` here,
        which is precisely the kind of mistake a translation-only fixture
        cannot catch.
        """
        parent_global = column_vector_transform((1.0, 0.0, 0.0), 90.0)
        child_global = column_vector_transform((3.0, 0.0, 0.0), 90.0)

        skeleton = build_skeleton(
            [-1, 0],
            ["parent", "child"],
            [parent_global.T, child_global.T],
            storage=MatrixStorage.ROW_VECTOR,
            conversion=IDENTITY_CONVERSION,
        )

        expected_local = np.linalg.inv(parent_global) @ child_global
        assert np.allclose(skeleton.bones[1].local_rest, expected_local)
        assert skeleton.bones[1].translation == pytest.approx(
            [0.0, -2.0, 0.0], abs=1e-12
        )
        assert skeleton.bones[1].rotation == pytest.approx([0.0, 0.0, 0.0, 1.0])

    def test_rest_skinning_matrices_are_identity(self):
        matrices = [
            row_vector_matrix((0.0, 0.0, 0.0)),
            row_vector_matrix((1.0, 2.0, 0.0), 30.0),
            row_vector_matrix((1.0, 5.0, 0.0), -45.0),
        ]

        skeleton = build_skeleton(
            [-1, 0, 1], ["a", "b", "c"], matrices, conversion=IDENTITY_CONVERSION
        )

        for matrix in skeleton.skinning_matrices():
            assert np.allclose(matrix, np.identity(4), atol=1e-12)

    def test_inverse_bind_is_the_inverse_of_the_global_rest(self):
        matrices = [row_vector_matrix((2.0, 1.0, 0.0), 25.0)]

        skeleton = build_skeleton(
            [-1], ["only"], matrices, conversion=IDENTITY_CONVERSION
        )

        bone = skeleton.bones[0]
        assert np.allclose(bone.global_rest @ bone.inverse_bind, np.identity(4))
        assert np.allclose(bone.global_rest, contract_from_storage(matrices[0]))

    def test_bone_order_does_not_change_the_result(self):
        """
        Reordering storage must not move a single bone.

        Same three bones, stored children-first. Every bone has to land in the
        same place, which is what a topological walk guarantees and an
        in-order walk does not.
        """
        origins = {"root": (0.0, 0.0, 0.0), "mid": (1.0, 1.0, 0.0), "tip": (1.0, 3.0, 0.0)}
        angles = {"root": 0.0, "mid": 20.0, "tip": -35.0}

        def build(order):
            index_of = {name: index for index, name in enumerate(order)}
            parent_name = {"root": None, "mid": "root", "tip": "mid"}
            parents = [
                -1 if parent_name[name] is None else index_of[parent_name[name]]
                for name in order
            ]
            matrices = [row_vector_matrix(origins[name], angles[name]) for name in order]
            return build_skeleton(
                parents, list(order), matrices, conversion=IDENTITY_CONVERSION
            )

        forward = build(["root", "mid", "tip"])
        reversed_order = build(["tip", "mid", "root"])

        for name in origins:
            a = next(bone for bone in forward.bones if bone.name == name)
            b = next(bone for bone in reversed_order.bones if bone.name == name)
            assert np.allclose(a.global_rest, b.global_rest)
            assert np.allclose(a.local_rest, b.local_rest)
            assert np.allclose(a.inverse_bind, b.inverse_bind)

    def test_multiple_roots_are_reported(self):
        matrices = [row_vector_matrix((0.0, 0.0, 0.0)), row_vector_matrix((5.0, 0.0, 0.0))]

        skeleton = build_skeleton([-1, -1], ["a", "b"], matrices)

        assert skeleton.roots == [0, 1]
        assert any("2 root bones" in note for note in skeleton.diagnostics)

    def test_a_parent_cycle_is_rejected(self):
        with pytest.raises(SkeletonError, match="cycle"):
            build_skeleton(
                [1, 2, 0],
                ["a", "b", "c"],
                [np.identity(4)] * 3,
            )

    def test_an_out_of_range_parent_is_rejected(self):
        with pytest.raises(SkeletonError, match="outside of"):
            build_skeleton([-1, 7], ["a", "b"], [np.identity(4)] * 2)

    def test_a_singular_matrix_is_rejected(self):
        with pytest.raises(SkeletonError, match="singular"):
            build_skeleton([-1], ["a"], [np.zeros((4, 4))])

    def test_mismatched_array_lengths_are_rejected(self):
        with pytest.raises(SkeletonError, match="must match"):
            build_skeleton([-1, 0], ["only_one"], [np.identity(4)] * 2)


class TestMatrixRole:
    def test_inverse_bind_input_produces_the_same_hierarchy(self):
        """
        Declaring the other role must invert the input, not the output.

        The same skeleton expressed as inverse bind matrices has to yield the
        same global rest transforms.
        """
        globals_ = [
            column_vector_transform((0.0, 0.0, 0.0)),
            column_vector_transform((1.0, 2.0, 0.0), 30.0),
        ]
        as_global = [matrix.T for matrix in globals_]
        as_inverse = [np.linalg.inv(matrix).T for matrix in globals_]

        from_global = build_skeleton(
            [-1, 0],
            ["a", "b"],
            as_global,
            storage=MatrixStorage.ROW_VECTOR,
            role=MatrixRole.GLOBAL_BIND,
            conversion=IDENTITY_CONVERSION,
        )
        from_inverse = build_skeleton(
            [-1, 0],
            ["a", "b"],
            as_inverse,
            storage=MatrixStorage.ROW_VECTOR,
            role=MatrixRole.INVERSE_BIND,
            conversion=IDENTITY_CONVERSION,
        )

        for a, b in zip(from_global.bones, from_inverse.bones):
            assert np.allclose(a.global_rest, b.global_rest)
            assert np.allclose(a.local_rest, b.local_rest)

    def test_local_bind_input_accumulates_down_the_chain(self):
        locals_ = [
            column_vector_transform((1.0, 0.0, 0.0), 90.0),
            column_vector_transform((2.0, 0.0, 0.0)),
        ]

        skeleton = build_skeleton(
            [-1, 0],
            ["a", "b"],
            [matrix.T for matrix in locals_],
            storage=MatrixStorage.ROW_VECTOR,
            role=MatrixRole.LOCAL_BIND,
            conversion=IDENTITY_CONVERSION,
        )

        # The child is 2 units along the parent's rotated X axis, so it ends up
        # 2 units along world Y from the parent origin at (1, 0, 0).
        assert skeleton.bones[1].global_rest[:3, 3] == pytest.approx([1.0, 2.0, 0.0])

    def test_bones_far_from_the_geometry_raise_a_diagnostic(self):
        matrices = [row_vector_matrix((0.0, 0.0, 0.0)), row_vector_matrix((5000.0, 0.0, 0.0))]

        skeleton = build_skeleton(
            [-1, 0],
            ["a", "b"],
            matrices,
            conversion=IDENTITY_CONVERSION,
            mesh_positions=[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
        )

        assert any("mesh radius" in note for note in skeleton.diagnostics)


class TestCoordinateConversion:
    def test_the_conversion_is_a_similarity_transform(self):
        """
        ``C @ M @ inverse(C)`` keeps the relationship between points.

        Converting a point and then transforming it must equal transforming it
        and then converting, which is the only way mesh and skeleton stay in
        the same space.
        """
        transform = column_vector_transform((1.0, 2.0, 3.0), 55.0)
        point = np.array([0.4, -1.2, 2.5, 1.0])

        converted_transform = NEOX_TO_GLTF.transform(transform)
        converted_point = NEOX_TO_GLTF.matrix @ point

        assert np.allclose(
            converted_transform @ converted_point,
            NEOX_TO_GLTF.matrix @ (transform @ point),
        )

    def test_the_flip_reverses_winding(self):
        assert NEOX_TO_GLTF.flips_winding is True
        assert IDENTITY_CONVERSION.flips_winding is False

    def test_converted_skeleton_still_rests_at_identity(self):
        matrices = [
            row_vector_matrix((0.0, 0.0, 0.0)),
            row_vector_matrix((1.0, 2.0, 0.0), 30.0),
        ]

        skeleton = build_skeleton([-1, 0], ["a", "b"], matrices, conversion=NEOX_TO_GLTF)

        for matrix in skeleton.skinning_matrices():
            assert np.allclose(matrix, np.identity(4), atol=1e-12)
        # The mirrored bone origin has to follow the mirrored geometry.
        assert skeleton.bones[1].global_rest[:3, 3] == pytest.approx([-1.0, 2.0, 0.0])
