"""
Tests for the shared glTF/GLB scene builder.

Every assertion reads the *exported file* through the independent reader in
``tests/support/gltf_reader.py``, never the builder's own intermediate state.
"""

import numpy as np
import pytest

from core.mesh_converter import gltf_scene
from core.mesh_converter.formats import glb, gltf
from core.mesh_converter.gltf_scene import (
    MeshExportError,
    SkinDataError,
    build_scene,
)
from core.mesh_converter.skeleton import IDENTITY_CONVERSION
from tests.support import gltf_spec_check as spec
from tests.support.gltf_reader import read_any
from tests.support.synthetic import (
    asymmetric_character,
    make_mesh_data,
    row_vector_matrix,
)


def export_and_read(mesh, **options):
    """Export as GLB and parse it back with the independent reader."""
    document = read_any(glb.convert(mesh, **options))
    assert spec.check_gltf(document) == []
    return document


class TestHierarchy:
    def test_every_bone_is_reachable_from_the_scene(self):
        """
        Nodes are created first and linked afterwards.

        The old exporter linked a child to its parent only if the parent node
        already existed, so any bone stored before its parent -- which the
        parser's trailing ``dummy_root`` guarantees -- fell out of the scene
        graph entirely.
        """
        mesh = asymmetric_character()
        document = export_and_read(mesh)
        gltf_data = document.json_data

        reachable = spec.reachable_nodes(gltf_data)
        joint_nodes = set(gltf_data["skins"][0]["joints"])

        assert joint_nodes <= reachable
        assert len(joint_nodes) == len(mesh.bones.names)

    def test_parent_links_follow_the_source_hierarchy(self):
        mesh = asymmetric_character()
        document = export_and_read(mesh)
        nodes = document.json_data["nodes"]
        joints = document.json_data["skins"][0]["joints"]

        parent_of_node = {}
        for index, node in enumerate(nodes):
            for child in node.get("children", []):
                parent_of_node[child] = index

        for source_index, source_parent in enumerate(mesh.bones.parents):
            node_index = joints[source_index]
            if source_parent == -1:
                # Roots hang off the armature container, not off a bone.
                assert nodes[parent_of_node[node_index]]["name"].endswith("Armature")
            else:
                assert parent_of_node[node_index] == joints[source_parent]

    def test_multiple_roots_share_the_armature_container(self):
        mesh = make_mesh_data(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[(0, 1, 2)],
            bone_parents=[-1, -1],
            bone_names=["root_a", "root_b"],
            bone_matrices=[
                row_vector_matrix((0.0, 0.0, 0.0)),
                row_vector_matrix((5.0, 0.0, 0.0)),
            ],
            joints=[(0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0)],
            weights=[(1.0, 0.0, 0.0, 0.0)] * 3,
        )

        document = export_and_read(mesh)
        gltf_data = document.json_data

        reachable = spec.reachable_nodes(gltf_data)
        assert set(gltf_data["skins"][0]["joints"]) <= reachable
        assert len(gltf_data["skins"][0]["joints"]) == 2


class TestSkinMapping:
    def test_joints_attribute_indexes_the_joint_palette(self):
        """
        ``JOINTS_0`` holds positions in ``skin.joints``, not node indices.

        The old exporter wrote node indices, shifted by the auxiliary nodes, so
        with two bones the values could reach 3 while the palette only had
        slots 0 and 1.
        """
        mesh = asymmetric_character()
        document = export_and_read(mesh)
        gltf_data = document.json_data

        primitive = gltf_data["meshes"][0]["primitives"][0]
        joints = document.accessor(primitive["attributes"]["JOINTS_0"])
        joint_count = len(gltf_data["skins"][0]["joints"])

        assert int(joints.max()) < joint_count
        # One vertex per bone, each bound to its own bone, so every slot is used.
        assert sorted(set(joints[:, 0].tolist())) == list(range(joint_count))

    def test_bone_255_keeps_its_slot_with_16_bit_indices(self):
        """
        255 is a real bone when the source indices are 16 bit.

        Only ``65535`` marks an empty slot at that width. The old exporter
        rewrote both to the root node index, silently moving the influence.
        """
        bone_count = 257
        parents = [-1] + list(range(bone_count - 1))
        mesh = make_mesh_data(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[(0, 1, 2)],
            bone_parents=parents,
            bone_names=[f"bone_{index}" for index in range(bone_count)],
            bone_matrices=[
                row_vector_matrix((float(index) * 0.01, 0.0, 0.0))
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

        document = export_and_read(mesh)
        primitive = document.json_data["meshes"][0]["primitives"][0]
        joints = document.accessor(primitive["attributes"]["JOINTS_0"])
        slot_to_node = document.json_data["skins"][0]["joints"]
        nodes = document.json_data["nodes"]

        assert int(joints[0, 0]) == 255
        assert nodes[slot_to_node[int(joints[0, 0])]]["name"] == "bone_255"
        assert nodes[slot_to_node[int(joints[1, 0])]]["name"] == "bone_256"

    def test_a_weighted_sentinel_is_refused(self):
        mesh = make_mesh_data(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[(0, 1, 2)],
            bone_parents=[-1, 0],
            bone_names=["root", "child"],
            bone_matrices=[row_vector_matrix((0.0, 0.0, 0.0))] * 2,
            joints=[(255, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
            weights=[(1.0, 0.0, 0.0, 0.0)] * 3,
        )

        with pytest.raises(SkinDataError, match="sentinel"):
            glb.convert(mesh)

    def test_an_out_of_range_joint_with_weight_is_refused(self):
        mesh = make_mesh_data(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[(0, 1, 2)],
            bone_parents=[-1, 0],
            bone_names=["root", "child"],
            bone_matrices=[row_vector_matrix((0.0, 0.0, 0.0))] * 2,
            joints=[(5, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
            weights=[(1.0, 0.0, 0.0, 0.0)] * 3,
            joint_index_bits=16,
        )

        with pytest.raises(SkinDataError, match="outside of"):
            glb.convert(mesh)

    def test_unused_slots_are_written_as_joint_zero_with_zero_weight(self):
        mesh = asymmetric_character()
        document = export_and_read(mesh)
        primitive = document.json_data["meshes"][0]["primitives"][0]
        joints = document.accessor(primitive["attributes"]["JOINTS_0"])
        weights = document.accessor(primitive["attributes"]["WEIGHTS_0"])

        assert np.all(joints[weights == 0.0] == 0)

    def test_a_drifting_weight_sum_is_refused_instead_of_renormalised(self):
        """A large drift means the influences were read at the wrong offset."""
        mesh = make_mesh_data(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[(0, 1, 2)],
            bone_parents=[-1, 0],
            bone_names=["root", "child"],
            bone_matrices=[
                row_vector_matrix((0.0, 0.0, 0.0)),
                row_vector_matrix((1.0, 0.0, 0.0)),
            ],
            joints=[(0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
            weights=[(0.25, 0.25, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
        )

        with pytest.raises(SkinDataError, match="wrong offset"):
            glb.convert(mesh)

    def test_a_tiny_drift_is_normalised(self):
        mesh = make_mesh_data(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[(0, 1, 2)],
            bone_parents=[-1, 0],
            bone_names=["root", "child"],
            bone_matrices=[
                row_vector_matrix((0.0, 0.0, 0.0)),
                row_vector_matrix((1.0, 0.0, 0.0)),
            ],
            joints=[(0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
            weights=[
                (0.5001, 0.5, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
            ],
        )

        document = export_and_read(mesh)
        primitive = document.json_data["meshes"][0]["primitives"][0]
        weights = document.accessor(primitive["attributes"]["WEIGHTS_0"])

        assert weights.sum(axis=1) == pytest.approx([1.0, 1.0, 1.0], abs=1e-7)

    def test_duplicate_joints_in_one_vertex_are_merged(self):
        mesh = make_mesh_data(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[(0, 1, 2)],
            bone_parents=[-1, 0],
            bone_names=["root", "child"],
            bone_matrices=[
                row_vector_matrix((0.0, 0.0, 0.0)),
                row_vector_matrix((1.0, 0.0, 0.0)),
            ],
            joints=[(1, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
            weights=[(0.3, 0.3, 0.4, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
        )

        document = export_and_read(mesh)
        primitive = document.json_data["meshes"][0]["primitives"][0]
        joints = document.accessor(primitive["attributes"]["JOINTS_0"])
        weights = document.accessor(primitive["attributes"]["WEIGHTS_0"])

        influences = {
            int(joints[0, slot]): float(weights[0, slot])
            for slot in range(4)
            if weights[0, slot] > 0.0
        }
        assert influences == pytest.approx({1: 0.6, 0: 0.4})

    def test_negative_weights_are_refused(self):
        mesh = make_mesh_data(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[(0, 1, 2)],
            bone_parents=[-1],
            bone_names=["root"],
            bone_matrices=[row_vector_matrix((0.0, 0.0, 0.0))],
            joints=[(0, 0, 0, 0)] * 3,
            weights=[(-1.0, 2.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
        )

        # Caught by rig validation before the influence pass; SkinDataError is
        # a MeshExportError, so either layer satisfies the contract as long as
        # the cause is named.
        with pytest.raises(MeshExportError, match="negative weight"):
            glb.convert(mesh)

    def test_partially_unskinned_geometry_is_refused(self):
        """Binding the leftovers to the root would hide the real cause."""
        mesh = make_mesh_data(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[(0, 1, 2)],
            bone_parents=[-1],
            bone_names=["root"],
            bone_matrices=[row_vector_matrix((0.0, 0.0, 0.0))],
            joints=[(0, 0, 0, 0)] * 3,
            weights=[
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
            ],
        )

        with pytest.raises(SkinDataError, match="no usable"):
            glb.convert(mesh)

    def test_bones_without_any_influence_export_without_a_skin(self):
        """
        Type 100 yields all-zero weights.

        A rig that deforms nothing must not be advertised as a skin, but the
        skeleton is still worth exporting.
        """
        mesh = make_mesh_data(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[(0, 1, 2)],
            bone_parents=[-1, 0],
            bone_names=["root", "child"],
            bone_matrices=[
                row_vector_matrix((0.0, 0.0, 0.0)),
                row_vector_matrix((1.0, 0.0, 0.0)),
            ],
            joints=[(0, 0, 0, 0)] * 3,
            weights=[(0.0, 0.0, 0.0, 0.0)] * 3,
            mesh_type=100,
        )

        document = export_and_read(mesh)
        gltf_data = document.json_data

        assert "skins" not in gltf_data
        assert "JOINTS_0" not in gltf_data["meshes"][0]["primitives"][0]["attributes"]
        # The skeleton still made it into the scene.
        node_names = {node["name"] for node in gltf_data["nodes"]}
        assert {"root", "child"} <= node_names


class TestInverseBindMatrices:
    def test_the_skin_references_its_inverse_bind_matrices(self):
        """
        The property used to be commented out, so importers assumed identity.

        A vertex at the origin bound to a bone translated two units then landed
        at (2, 0, 0) instead of staying put.
        """
        mesh = asymmetric_character()
        document = export_and_read(mesh)
        skin = document.json_data["skins"][0]

        assert "inverseBindMatrices" in skin
        matrices = document.accessor(skin["inverseBindMatrices"])
        assert len(matrices) == len(skin["joints"])

    def test_inverse_bind_matrices_are_in_joint_slot_order(self):
        mesh = asymmetric_character()
        document = export_and_read(mesh)
        gltf_data = document.json_data
        skin = gltf_data["skins"][0]
        matrices = document.accessor(skin["inverseBindMatrices"])

        scene = build_scene(mesh)
        for slot, node_index in enumerate(skin["joints"]):
            name = gltf_data["nodes"][node_index]["name"]
            bone = next(bone for bone in scene.skeleton.bones if bone.name == name)
            assert np.allclose(matrices[slot], bone.inverse_bind, atol=1e-5)

    def test_matrices_are_serialised_column_major(self):
        """
        glTF stores matrices column-major; the translation lands in elements
        12, 13, 14 of the flat array.
        """
        mesh = make_mesh_data(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[(0, 1, 2)],
            bone_parents=[-1],
            bone_names=["root"],
            bone_matrices=[row_vector_matrix((2.0, 3.0, 4.0))],
            joints=[(0, 0, 0, 0)] * 3,
            weights=[(1.0, 0.0, 0.0, 0.0)] * 3,
        )

        scene = build_scene(mesh, conversion=IDENTITY_CONVERSION)
        flat = gltf_scene._serialize_column_major(scene.skeleton.bones[0].global_rest)

        assert flat[12:15] == pytest.approx([2.0, 3.0, 4.0])
        assert flat[15] == pytest.approx(1.0)


class TestBoneNodes:
    def test_bone_nodes_use_trs_so_they_can_be_animated(self):
        """glTF animation channels can only target translation/rotation/scale."""
        mesh = asymmetric_character()
        document = export_and_read(mesh)
        gltf_data = document.json_data
        joints = set(gltf_data["skins"][0]["joints"])

        for node_index in joints:
            node = gltf_data["nodes"][node_index]
            assert "matrix" not in node, f"{node['name']} was baked into a matrix"

    def test_trs_reproduces_the_local_rest_transform(self):
        from tests.support.skinning import node_local_matrix

        mesh = asymmetric_character()
        document = export_and_read(mesh)
        gltf_data = document.json_data
        scene = build_scene(mesh)

        for node_index in gltf_data["skins"][0]["joints"]:
            node = gltf_data["nodes"][node_index]
            bone = next(bone for bone in scene.skeleton.bones if bone.name == node["name"])
            assert np.allclose(node_local_matrix(node), bone.local_rest, atol=1e-6)

    def test_a_sheared_bone_falls_back_to_a_baked_matrix(self):
        sheared = np.identity(4)
        sheared[0, 1] = 0.5
        mesh = make_mesh_data(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[(0, 1, 2)],
            bone_parents=[-1],
            bone_names=["sheared"],
            bone_matrices=[sheared],
            joints=[(0, 0, 0, 0)] * 3,
            weights=[(1.0, 0.0, 0.0, 0.0)] * 3,
        )

        document = export_and_read(mesh)
        node = next(
            node for node in document.json_data["nodes"] if node["name"] == "sheared"
        )

        assert "matrix" in node
        scene = build_scene(mesh)
        assert any("shear" in note for note in scene.diagnostics)


class TestGeometry:
    def test_geometry_is_not_transformed_twice(self):
        """
        Positions carry the basis change; nodes stay at identity.

        The skinned mesh node has no transform at all (glTF ignores it), and
        the armature container is identity, so the exported coordinates are the
        final ones.
        """
        mesh = asymmetric_character()
        document = export_and_read(mesh)
        gltf_data = document.json_data

        primitive = gltf_data["meshes"][0]["primitives"][0]
        positions = document.accessor(primitive["attributes"]["POSITION"])

        expected = np.asarray(mesh.mesh.position, dtype=np.float64)
        expected[:, 0] *= -1.0
        assert np.allclose(positions, expected, atol=1e-6)

        mesh_node = gltf_data["nodes"][0]
        assert not {"matrix", "translation", "rotation", "scale"} & set(mesh_node)

    def test_the_mirror_reverses_triangle_winding(self):
        mesh = asymmetric_character()
        document = export_and_read(mesh)
        primitive = document.json_data["meshes"][0]["primitives"][0]
        indices = document.accessor(primitive["indices"]).reshape(-1, 3)

        source = np.asarray(mesh.mesh.face, dtype=np.int64)
        assert np.array_equal(indices, source[:, [0, 2, 1]])

    def test_the_identity_conversion_keeps_the_source_winding(self):
        mesh = asymmetric_character()
        document = read_any(glb.convert(mesh, conversion=IDENTITY_CONVERSION))
        primitive = document.json_data["meshes"][0]["primitives"][0]
        indices = document.accessor(primitive["indices"]).reshape(-1, 3)

        assert np.array_equal(indices, np.asarray(mesh.mesh.face, dtype=np.int64))

    def test_normals_are_unit_length_and_follow_the_conversion(self):
        mesh = make_mesh_data(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[(0, 1, 2)],
            normals=[(3.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 5.0)],
        )

        document = export_and_read(mesh)
        primitive = document.json_data["meshes"][0]["primitives"][0]
        normals = document.accessor(primitive["attributes"]["NORMAL"])

        assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)
        assert np.allclose(normals[0], (-1.0, 0.0, 0.0), atol=1e-6)

    def test_a_mesh_without_bones_still_exports(self):
        mesh = make_mesh_data(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[(0, 1, 2)],
        )

        document = export_and_read(mesh)
        gltf_data = document.json_data

        assert "skins" not in gltf_data
        assert len(gltf_data["nodes"]) == 1
        assert gltf_data["scenes"][0]["nodes"] == [0]

    def test_extra_uv_layers_are_truncated_to_the_vertex_count(self):
        mesh = make_mesh_data(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[(0, 1, 2)],
            uvs=[(0.1, 0.2), (0.3, 0.4), (0.5, 0.6), (0.7, 0.8), (0.9, 1.0), (0.0, 0.0)],
        )

        document = export_and_read(mesh)
        primitive = document.json_data["meshes"][0]["primitives"][0]
        uvs = document.accessor(primitive["attributes"]["TEXCOORD_0"])

        assert len(uvs) == 3
        assert np.allclose(uvs, [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)], atol=1e-6)

    def test_inconsistent_geometry_is_refused(self):
        mesh = make_mesh_data(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[(0, 1, 9)],
        )

        with pytest.raises(MeshExportError, match="geometry is inconsistent"):
            glb.convert(mesh)


class TestGltfAndGlbAgree:
    def test_both_writers_emit_the_same_scene(self):
        mesh = asymmetric_character()

        gltf_document = read_any(gltf.convert(mesh))
        glb_document = read_any(glb.convert(mesh))

        gltf_json = dict(gltf_document.json_data)
        glb_json = dict(glb_document.json_data)
        # The only intended difference is how the buffer is delivered.
        gltf_buffers = gltf_json.pop("buffers")
        glb_buffers = glb_json.pop("buffers")

        assert gltf_json == glb_json
        assert gltf_buffers[0]["byteLength"] == glb_buffers[0]["byteLength"]
        assert "uri" in gltf_buffers[0]
        assert "uri" not in glb_buffers[0]
        assert gltf_document.buffers[0] == glb_document.buffers[0]
        assert spec.check_gltf(gltf_document) == []
