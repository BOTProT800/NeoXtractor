"""
Tests for the ``.glb`` binary container.

The container is checked field by field against the specification layout, and
then read back with the independent reader and -- when available -- with
``pygltflib``, a third-party implementation that shares nothing with this
project.
"""

import io
import json
import struct

import numpy as np
import pytest

from core.mesh_converter import FORMATS
from core.mesh_converter.formats import glb
from tests.support import gltf_spec_check as spec
from tests.support.gltf_reader import GLTFReadError, read_glb
from tests.support.synthetic import asymmetric_character, make_mesh_data


def test_glb_is_registered_in_the_save_menu():
    names = [module.NAME for module in FORMATS]
    extensions = [module.EXTENSION for module in FORMATS]

    assert glb in FORMATS
    assert ".glb" in extensions
    assert any("GLB" in name for name in names)
    # The JSON form stays available for inspection.
    assert ".gltf" in extensions


class TestContainerLayout:
    def test_header_fields_match_the_specification(self):
        payload = glb.convert(asymmetric_character())

        magic, version, total = struct.unpack_from("<III", payload, 0)

        assert payload[:4] == b"glTF"
        assert magic == 0x46546C67
        assert version == 2
        assert total == len(payload)

    def test_chunks_are_json_then_bin_and_padded_to_four_bytes(self):
        payload = glb.convert(asymmetric_character())

        offset = 12
        chunks = []
        while offset < len(payload):
            length, chunk_type = struct.unpack_from("<II", payload, offset)
            offset += 8
            chunks.append((chunk_type, length, payload[offset : offset + length]))
            offset += length

        assert offset == len(payload)
        assert [chunk_type for chunk_type, _, _ in chunks] == [0x4E4F534A, 0x004E4942]
        for chunk_type, length, data in chunks:
            assert length % 4 == 0, "chunk lengths must be 4 byte aligned"
            assert len(data) == length

        json_data = chunks[0][2]
        # JSON is space padded so it stays parseable as text.
        assert json_data.rstrip(b"\x20")[-1:] == b"}"
        json.loads(json_data.decode("utf-8"))

    def test_the_buffer_has_no_uri_and_matches_the_bin_chunk(self):
        payload = glb.convert(asymmetric_character())
        document = read_glb(payload)

        buffer = document.json_data["buffers"][0]
        assert "uri" not in buffer
        assert len(document.buffers[0]) >= buffer["byteLength"]

    def test_everything_is_written_little_endian(self):
        """A big-endian header would decode as a nonsensical length."""
        payload = glb.convert(asymmetric_character())

        little = struct.unpack_from("<I", payload, 8)[0]
        big = struct.unpack_from(">I", payload, 8)[0]

        assert little == len(payload)
        assert big != len(payload)

    def test_pack_glb_pads_an_unaligned_payload(self):
        packed = glb.pack_glb(b"{}", b"\x01\x02\x03")

        _, _, total = struct.unpack_from("<III", packed, 0)
        json_length = struct.unpack_from("<I", packed, 12)[0]
        bin_length = struct.unpack_from("<I", packed, 12 + 8 + json_length)[0]

        assert total == len(packed)
        assert json_length % 4 == 0
        assert bin_length == 4  # 3 bytes zero-padded

    def test_a_corrupted_length_is_rejected_by_the_reader(self):
        payload = bytearray(glb.convert(asymmetric_character()))
        struct.pack_into("<I", payload, 8, len(payload) + 4)

        with pytest.raises(GLTFReadError, match="header declares"):
            read_glb(bytes(payload))


class TestRoundTrip:
    def test_the_binary_reads_back_as_the_same_scene(self):
        mesh = asymmetric_character()
        document = read_glb(glb.convert(mesh))

        assert spec.check_gltf(document) == []
        primitive = document.json_data["meshes"][0]["primitives"][0]
        positions = document.accessor(primitive["attributes"]["POSITION"])

        expected = np.asarray(mesh.mesh.position, dtype=np.float64)
        expected[:, 0] *= -1.0
        assert np.allclose(positions, expected, atol=1e-6)

    def test_packaging_does_not_alter_the_rig(self):
        """Same skin data whether it travels as base64 or as a BIN chunk."""
        from core.mesh_converter.formats import gltf
        from tests.support.gltf_reader import read_gltf

        mesh = asymmetric_character()
        binary = read_glb(glb.convert(mesh))
        text = read_gltf(gltf.convert(mesh))

        assert binary.json_data["skins"] == text.json_data["skins"]
        for accessor_index in range(len(binary.json_data["accessors"])):
            assert np.allclose(
                binary.accessor(accessor_index).astype(np.float64),
                text.accessor(accessor_index).astype(np.float64),
            )

    def test_a_mesh_without_bones_produces_a_valid_glb(self):
        mesh = make_mesh_data(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[(0, 1, 2)],
        )

        document = read_glb(glb.convert(mesh))

        assert spec.check_gltf(document) == []
        assert "skins" not in document.json_data


class TestTrimesh:
    """Cross-check with trimesh, a second unrelated glTF implementation."""

    def test_trimesh_loads_the_scene_and_its_graph(self):
        trimesh = pytest.importorskip("trimesh")

        mesh = asymmetric_character()
        scene = trimesh.load(
            io.BytesIO(glb.convert(mesh)), file_type="glb"
        )

        assert list(scene.geometry) == ["NeoXMesh"]
        geometry = scene.geometry["NeoXMesh"]
        assert geometry.vertices.shape == (len(mesh.mesh.position), 3)
        assert geometry.faces.shape == (len(mesh.mesh.face), 3)
        # Mesh node, armature container and one node per bone, plus the world
        # frame trimesh adds itself.
        assert len(scene.graph.nodes) >= 2 + len(mesh.bones.names)

    def test_trimesh_decodes_the_same_positions(self):
        trimesh = pytest.importorskip("trimesh")

        mesh = asymmetric_character()
        scene = trimesh.load(
            io.BytesIO(glb.convert(mesh)), file_type="glb"
        )
        vertices = scene.geometry["NeoXMesh"].vertices

        expected = np.asarray(mesh.mesh.position, dtype=np.float64)
        expected[:, 0] *= -1.0
        assert np.allclose(vertices, expected, atol=1e-6)

    def test_trimesh_sees_every_bone_node(self):
        trimesh = pytest.importorskip("trimesh")

        mesh = asymmetric_character()
        scene = trimesh.load(
            io.BytesIO(glb.convert(mesh)), file_type="glb"
        )

        assert set(mesh.bones.names) <= set(scene.graph.nodes)


class TestThirdPartyReader:
    """Cross-check with pygltflib, an unrelated implementation."""

    def test_pygltflib_parses_the_container(self):
        pygltflib = pytest.importorskip("pygltflib")

        mesh = asymmetric_character()
        payload = glb.convert(mesh)
        parsed = pygltflib.GLTF2.load_from_bytes(payload)

        assert parsed.asset.version == "2.0"
        assert len(parsed.skins) == 1
        assert len(parsed.skins[0].joints) == len(mesh.bones.names)
        assert parsed.skins[0].inverseBindMatrices is not None
        assert parsed.nodes[parsed.scenes[parsed.scene].nodes[0]].skin == 0

    def test_pygltflib_decodes_the_positions(self):
        pygltflib = pytest.importorskip("pygltflib")

        mesh = asymmetric_character()
        parsed = pygltflib.GLTF2.load_from_bytes(glb.convert(mesh))
        blob = parsed.binary_blob()

        primitive = parsed.meshes[0].primitives[0]
        accessor = parsed.accessors[primitive.attributes.POSITION]
        view = parsed.bufferViews[accessor.bufferView]
        start = (view.byteOffset or 0) + (accessor.byteOffset or 0)
        decoded = np.frombuffer(
            blob, dtype=np.float32, count=accessor.count * 3, offset=start
        ).reshape(-1, 3)

        expected = np.asarray(mesh.mesh.position, dtype=np.float32)
        expected[:, 0] *= -1.0
        assert np.allclose(decoded, expected, atol=1e-6)
