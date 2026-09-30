"""
Deformation acceptance tests.

The mesh is exported, read back with the independent reader, and skinned on the
CPU using the glTF skinning equation. Rest pose, root motion, independent
branches and left/right separation are each checked against positions worked
out from the fixture's own geometry.

Rest-pose identity alone proves nothing -- deriving a matrix and its inverse
from the same wrong interpretation satisfies it -- so every test here moves a
bone and checks *where* the vertices went.
"""

import numpy as np
import pytest

from core.mesh_converter.formats import glb
from tests.support import gltf_spec_check as spec
from tests.support.gltf_reader import read_any
from tests.support.skinning import (
    node_local_matrix,
    rotation_override,
    skin_vertices,
)
from tests.support.synthetic import asymmetric_character, make_mesh_data, row_vector_matrix

#: Relative tolerance on reconstructed positions, expressed as a fraction of
#: the mesh diagonal. float32 storage is the limiting factor here.
POSITION_TOLERANCE_RATIO = 1e-5


@pytest.fixture
def character():
    """Export the asymmetric rig and hand back everything the tests need."""
    mesh = asymmetric_character()
    document = read_any(glb.convert(mesh))
    assert spec.check_gltf(document) == []

    gltf_data = document.json_data
    node_index_of = {
        gltf_data["nodes"][node]["name"]: node
        for node in gltf_data["skins"][0]["joints"]
    }
    positions = np.asarray(mesh.mesh.position, dtype=np.float64)
    positions[:, 0] *= -1.0
    diagonal = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))

    return {
        "mesh": mesh,
        "document": document,
        "gltf": gltf_data,
        "node_index_of": node_index_of,
        "vertex_of": {name: index for index, name in enumerate(mesh.bones.names)},
        "rest": positions,
        "tolerance": diagonal * POSITION_TOLERANCE_RATIO,
    }


def bone_origin(gltf_data, node_index_of, name):
    """Resolve a bone's rest origin in scene space from the file alone."""
    from tests.support.skinning import global_node_matrices

    globals_ = global_node_matrices(gltf_data)
    return globals_[node_index_of[name]][:3, 3]


class TestPivots:
    def test_bone_origins_match_the_rig_layout(self, character):
        """
        Pivots are checked directly, not through the rest pose.

        The old exporter flattened each bone matrix so that glTF read the
        translation as the last *row*, which is the affine ``(0, 0, 0, 1)``
        slot: every bone collapsed onto the model origin. The rest pose still
        came out pixel perfect -- identity inverse bind matrices against
        identity globals -- while every joint pivot was wrong, so any bone
        rotation turned the mesh about the origin instead of the joint.

        This is the check that would have caught the reported deformation.
        """
        gltf_data = character["gltf"]
        node_index_of = character["node_index_of"]
        mesh = character["mesh"]

        for source_index, name in enumerate(mesh.bones.names):
            # The file stores the origin in the last row (row-vector layout);
            # export mirrors X.
            stored = np.asarray(mesh.bones.matrix[source_index], dtype=np.float64)
            expected = stored[3, :3].copy()
            expected[0] *= -1.0

            actual = bone_origin(gltf_data, node_index_of, name)
            assert np.allclose(actual, expected, atol=character["tolerance"]), (
                f"{name} pivot is at {actual.tolist()}, expected {expected.tolist()}"
            )

        # Guard against the degenerate case the old exporter produced, where
        # every origin was (0, 0, 0) and the assertions above would only pass
        # for a rig that genuinely sits at the origin.
        origins = np.array(
            [bone_origin(gltf_data, node_index_of, name) for name in mesh.bones.names]
        )
        assert np.abs(origins).max() > 0.5, "every bone collapsed onto the origin"


class TestRestPose:
    def test_the_rest_pose_reproduces_the_source_geometry(self, character):
        """
        With the rig at rest every skinning matrix is identity.

        This is the check that failed hardest before: without
        ``inverseBindMatrices`` the importer assumed identity, and a vertex at
        the origin bound to a bone two units away landed at (2, 0, 0).
        """
        skinned = skin_vertices(character["document"])

        assert np.allclose(skinned, character["rest"], atol=character["tolerance"])

    def test_returning_to_rest_recovers_the_geometry(self, character):
        gltf_data = character["gltf"]
        node_index = character["node_index_of"]["root"]
        node = gltf_data["nodes"][node_index]

        posed = skin_vertices(
            character["document"], {node_index: rotation_override(node, 47.0)}
        )
        back_at_rest = skin_vertices(
            character["document"], {node_index: node_local_matrix(node)}
        )

        assert not np.allclose(posed, character["rest"], atol=character["tolerance"])
        assert np.allclose(back_at_rest, character["rest"], atol=character["tolerance"])


class TestRootMotion:
    def test_translating_the_root_moves_the_whole_mesh_rigidly(self, character):
        gltf_data = character["gltf"]
        node_index = character["node_index_of"]["root"]
        offset = np.array([3.0, -2.0, 1.5])

        local = node_local_matrix(gltf_data["nodes"][node_index]).copy()
        local[:3, 3] += offset

        skinned = skin_vertices(character["document"], {node_index: local})

        assert np.allclose(
            skinned, character["rest"] + offset, atol=character["tolerance"]
        )

    def test_rotating_the_root_turns_everything_about_its_own_pivot(self, character):
        gltf_data = character["gltf"]
        node_index = character["node_index_of"]["root"]
        node = gltf_data["nodes"][node_index]
        pivot = bone_origin(gltf_data, character["node_index_of"], "root")

        skinned = skin_vertices(
            character["document"], {node_index: rotation_override(node, 90.0, "z")}
        )

        angle = np.radians(90.0)
        cos, sin = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
        expected = (character["rest"] - pivot) @ rotation.T + pivot

        assert np.allclose(skinned, expected, atol=character["tolerance"])

    def test_rotation_preserves_distances(self, character):
        """A rig error usually shows up as stretching, not just displacement."""
        gltf_data = character["gltf"]
        node_index = character["node_index_of"]["root"]
        node = gltf_data["nodes"][node_index]

        skinned = skin_vertices(
            character["document"], {node_index: rotation_override(node, 33.0, "y")}
        )

        rest = character["rest"]
        for a in range(len(rest)):
            for b in range(a + 1, len(rest)):
                assert np.linalg.norm(skinned[a] - skinned[b]) == pytest.approx(
                    np.linalg.norm(rest[a] - rest[b]), abs=character["tolerance"]
                )


class TestIndependentBranches:
    def test_rotating_one_arm_leaves_the_other_untouched(self, character):
        """
        Each side is driven separately so a left/right swap cannot hide.

        The fixture is deliberately asymmetric: the left arm sits at
        x = 1 with its forearm 2 units up, the right arm at x = -3 with its
        forearm 3 units up.
        """
        gltf_data = character["gltf"]
        node_index_of = character["node_index_of"]
        vertex_of = character["vertex_of"]
        rest = character["rest"]
        tolerance = character["tolerance"]

        for moved, still in (("arm_l", "arm_r"), ("arm_r", "arm_l")):
            node_index = node_index_of[moved]
            node = gltf_data["nodes"][node_index]
            skinned = skin_vertices(
                character["document"], {node_index: rotation_override(node, 60.0, "z")}
            )

            moved_vertex = vertex_of[moved]
            still_vertex = vertex_of[still]
            forearm_moved = vertex_of[f"forearm_{moved[-1]}"]
            forearm_still = vertex_of[f"forearm_{still[-1]}"]

            assert not np.allclose(
                skinned[moved_vertex], rest[moved_vertex], atol=tolerance
            ), f"{moved} did not move"
            assert not np.allclose(
                skinned[forearm_moved], rest[forearm_moved], atol=tolerance
            ), f"the child of {moved} did not follow"
            assert np.allclose(
                skinned[still_vertex], rest[still_vertex], atol=tolerance
            ), f"{still} moved although it was not touched"
            assert np.allclose(
                skinned[forearm_still], rest[forearm_still], atol=tolerance
            ), f"the child of {still} moved although it was not touched"
            assert np.allclose(
                skinned[vertex_of["root"]], rest[vertex_of["root"]], atol=tolerance
            ), "the root vertex followed a child bone"

    def test_a_rotated_arm_turns_its_descendants_about_its_own_pivot(self, character):
        gltf_data = character["gltf"]
        node_index_of = character["node_index_of"]
        rest = character["rest"]

        for side in ("l", "r"):
            arm = f"arm_{side}"
            node_index = node_index_of[arm]
            node = gltf_data["nodes"][node_index]
            pivot = bone_origin(gltf_data, node_index_of, arm)

            skinned = skin_vertices(
                character["document"], {node_index: rotation_override(node, 90.0, "z")}
            )

            angle = np.radians(90.0)
            cos, sin = np.cos(angle), np.sin(angle)
            rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])

            for name in (arm, f"forearm_{side}"):
                vertex = character["vertex_of"][name]
                expected = rotation @ (rest[vertex] - pivot) + pivot
                assert np.allclose(
                    skinned[vertex], expected, atol=character["tolerance"]
                ), f"{name} did not turn about the {arm} pivot"

    def test_moving_a_leaf_does_not_disturb_its_parent(self, character):
        gltf_data = character["gltf"]
        node_index = character["node_index_of"]["forearm_l"]
        node = gltf_data["nodes"][node_index]

        skinned = skin_vertices(
            character["document"], {node_index: rotation_override(node, 75.0, "x")}
        )

        rest = character["rest"]
        tolerance = character["tolerance"]
        vertex_of = character["vertex_of"]

        assert not np.allclose(
            skinned[vertex_of["forearm_l"]], rest[vertex_of["forearm_l"]], atol=tolerance
        )
        for untouched in ("arm_l", "root", "arm_r", "forearm_r"):
            assert np.allclose(
                skinned[vertex_of[untouched]], rest[vertex_of[untouched]], atol=tolerance
            ), f"{untouched} moved although only its descendant was posed"


class TestBlendedWeights:
    def test_a_half_and_half_vertex_lands_between_both_bones(self):
        """
        Weighted blending is checked with a known split.

        The vertex sits at the origin, bound 60/40 to a bone that translates by
        (4, 0, 0) and one that stays put, so the answer is (2.4, 0, 0) in
        source space -- mirrored to (-2.4, 0, 0) on export.
        """
        mesh = make_mesh_data(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[(0, 1, 2)],
            bone_parents=[-1, 0],
            bone_names=["root", "mover"],
            bone_matrices=[
                row_vector_matrix((0.0, 0.0, 0.0)),
                row_vector_matrix((0.0, 0.0, 0.0)),
            ],
            joints=[(1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
            weights=[(0.6, 0.4, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
        )

        document = read_any(glb.convert(mesh))
        gltf_data = document.json_data
        node_index_of = {
            gltf_data["nodes"][node]["name"]: node
            for node in gltf_data["skins"][0]["joints"]
        }

        mover_index = node_index_of["mover"]
        local = node_local_matrix(gltf_data["nodes"][mover_index]).copy()
        local[:3, 3] += np.array([4.0, 0.0, 0.0])

        skinned = skin_vertices(document, {mover_index: local})

        assert skinned[0] == pytest.approx([2.4, 0.0, 0.0], abs=1e-5)
        assert skinned[1] == pytest.approx([-1.0, 0.0, 0.0], abs=1e-5)


def _random_rig(seed=7, bone_count=200, vertex_count=600, face_count=300):
    """A deep, randomly ordered rig with multi-influence vertices."""
    rng = np.random.default_rng(seed)

    # Build the tree in a random order so parents land before and after their
    # children in the arrays.
    order = rng.permutation(bone_count)
    parents = [-1] * bone_count
    for position, bone in enumerate(order):
        if position:
            parents[bone] = int(order[rng.integers(0, position)])

    matrices = [
        row_vector_matrix(
            rng.normal(0.0, 3.0, 3),
            float(rng.uniform(-180.0, 180.0)),
            "xyz"[rng.integers(0, 3)],
        )
        for _ in range(bone_count)
    ]

    positions = rng.normal(0.0, 5.0, (vertex_count, 3))
    joints, weights = [], []
    for _ in range(vertex_count):
        influences = int(rng.integers(1, 5))
        chosen = rng.choice(bone_count, size=influences, replace=False)
        raw = rng.random(influences)
        raw = raw / raw.sum()
        joints.append(
            tuple(int(value) for value in list(chosen) + [0] * (4 - influences))
        )
        weights.append(
            tuple(float(value) for value in list(raw) + [0.0] * (4 - influences))
        )

    faces = [
        tuple(int(value) for value in rng.integers(0, vertex_count, 3))
        for _ in range(face_count)
    ]

    mesh = make_mesh_data(
        positions=[tuple(p) for p in positions],
        faces=faces,
        bone_parents=parents,
        bone_names=[f"bone_{index}" for index in range(bone_count)],
        bone_matrices=matrices,
        joints=joints,
        weights=weights,
        joint_index_bits=16,
        mesh_type=5,
    )
    return mesh, parents


class TestLargeRandomRig:
    """A rig big and tangled enough that an ordering bug cannot hide."""

    def test_a_deep_randomly_ordered_rig_exports_and_rests_correctly(self):
        mesh, _ = _random_rig()
        document = read_any(glb.convert(mesh))

        assert spec.check_gltf(document) == []
        assert set(document.json_data["skins"][0]["joints"]) <= spec.reachable_nodes(
            document.json_data
        )

        skinned = skin_vertices(document)
        expected = np.asarray(mesh.mesh.position, dtype=np.float64)
        expected[:, 0] *= -1.0
        diagonal = float(np.linalg.norm(expected.max(axis=0) - expected.min(axis=0)))

        assert np.abs(skinned - expected).max() < diagonal * POSITION_TOLERANCE_RATIO

    def test_only_the_vertices_influenced_by_a_subtree_move(self):
        """
        Rotate one mid-chain bone and check the exact set of movers.

        A vertex may move only if one of its influencing bones is inside the
        rotated bone's subtree. Anything else moving means weights or parent
        links landed on the wrong bone.
        """
        mesh, parents = _random_rig()
        document = read_any(glb.convert(mesh))
        gltf_data = document.json_data
        slot_to_node = gltf_data["skins"][0]["joints"]

        # Pick the bone with the largest subtree that is not a root.
        descendants = {index: {index} for index in range(len(parents))}
        for index in range(len(parents)):
            cursor = parents[index]
            while cursor != -1:
                descendants[cursor].add(index)
                cursor = parents[cursor]
        target = max(
            (index for index in range(len(parents)) if parents[index] != -1),
            key=lambda index: len(descendants[index]),
        )
        subtree = descendants[target]

        node_index = slot_to_node[target]
        node = gltf_data["nodes"][node_index]
        skinned = skin_vertices(
            document, {node_index: rotation_override(node, 40.0, "y")}
        )

        rest = np.asarray(mesh.mesh.position, dtype=np.float64)
        rest[:, 0] *= -1.0
        diagonal = float(np.linalg.norm(rest.max(axis=0) - rest.min(axis=0)))
        tolerance = diagonal * POSITION_TOLERANCE_RATIO

        moved = np.abs(skinned - rest).max(axis=1) > tolerance
        for vertex in range(len(rest)):
            influenced = any(
                bone in subtree
                for bone, weight in zip(mesh.bones.joints[vertex], mesh.bones.weights[vertex])
                if weight > 0.0
            )
            if not influenced:
                assert not moved[vertex], (
                    f"vertex {vertex} moved without any influence in the subtree of "
                    f"bone {target}"
                )

        assert moved.any(), "rotating a large subtree moved nothing at all"


class TestSixteenBitVariant:
    def test_the_16_bit_fixture_deforms_the_same_way(self):
        """The index width must not change where a vertex ends up."""
        eight = asymmetric_character(joint_index_bits=8)
        sixteen = asymmetric_character(joint_index_bits=16)

        documents = [read_any(glb.convert(mesh)) for mesh in (eight, sixteen)]
        results = []
        for document in documents:
            gltf_data = document.json_data
            node_index_of = {
                gltf_data["nodes"][node]["name"]: node
                for node in gltf_data["skins"][0]["joints"]
            }
            node_index = node_index_of["arm_r"]
            node = gltf_data["nodes"][node_index]
            results.append(
                skin_vertices(document, {node_index: rotation_override(node, 45.0)})
            )

        assert np.allclose(results[0], results[1], atol=1e-6)
