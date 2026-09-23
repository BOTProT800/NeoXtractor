"""
glTF 2.0 binary container (``.glb``).

Same scene as the ``.gltf`` writer -- see
:mod:`core.mesh_converter.gltf_scene` -- packed into the binary container
instead of carrying its buffer as a base64 data URI. Packaging never touches
the rig.

Container layout, all fields little-endian 32-bit
(`glTF 2.0 spec, Binary glTF layout
<https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#binary-gltf-layout>`_)::

    header  magic 'glTF' | version 2 | total length
    chunk 0 length | type 'JSON' | JSON payload padded with spaces to 4 bytes
    chunk 1 length | type 'BIN'  | buffer payload padded with zeros to 4 bytes
"""

import json
import struct

from core.mesh_converter.gltf_scene import (
    MeshExportError,
    SkinDataError,
    build_scene,
)
from core.mesh_loader import MeshData

NAME = "glTF 2.0 Binary (GLB) Format"
EXTENSION = ".glb"

#: ASCII 'glTF', little-endian.
GLB_MAGIC = 0x46546C67
GLB_VERSION = 2
GLB_HEADER_SIZE = 12
GLB_CHUNK_HEADER_SIZE = 8

#: ASCII 'JSON'.
CHUNK_TYPE_JSON = 0x4E4F534A
#: ASCII 'BIN\0'.
CHUNK_TYPE_BIN = 0x004E4942

__all__ = [
    "NAME",
    "EXTENSION",
    "convert",
    "pack_glb",
    "GLB_MAGIC",
    "GLB_VERSION",
    "CHUNK_TYPE_JSON",
    "CHUNK_TYPE_BIN",
    "MeshExportError",
    "SkinDataError",
]


def _pad(payload: bytes, filler: bytes) -> bytes:
    """Pad a chunk payload to a 4 byte boundary."""
    remainder = (-len(payload)) % 4
    return payload + filler * remainder if remainder else payload


def pack_glb(json_bytes: bytes, binary: bytes = b"") -> bytes:
    """
    Wrap a JSON structure and its buffer in the GLB container.

    Parameters:
    - json_bytes: the serialised glTF JSON.
    - binary: the buffer contents, or empty for a JSON-only GLB.

    Returns:
    - bytes: the complete ``.glb`` file.
    """
    # The spec asks for space padding on JSON and zero padding on BIN so that
    # the JSON chunk stays parseable as text.
    json_chunk = _pad(json_bytes, b"\x20")
    binary_chunk = _pad(binary, b"\x00")

    total = GLB_HEADER_SIZE + GLB_CHUNK_HEADER_SIZE + len(json_chunk)
    if binary_chunk:
        total += GLB_CHUNK_HEADER_SIZE + len(binary_chunk)

    out = bytearray()
    out += struct.pack("<III", GLB_MAGIC, GLB_VERSION, total)
    out += struct.pack("<II", len(json_chunk), CHUNK_TYPE_JSON)
    out += json_chunk
    if binary_chunk:
        out += struct.pack("<II", len(binary_chunk), CHUNK_TYPE_BIN)
        out += binary_chunk

    if len(out) != total:  # pragma: no cover - guards the header arithmetic
        raise ValueError(
            f"GLB header declares {total} bytes but {len(out)} were written"
        )
    return bytes(out)


def convert(mesh: MeshData, **options) -> bytes:
    """
    Convert mesh to binary glTF format.

    Parameters:
    - mesh: MeshData object containing bones, vertices, faces, etc.
    - options: forwarded to :func:`core.mesh_converter.gltf_scene.build_scene`
      (``name``, ``conversion``, ``matrix_storage``, ``matrix_role``,
      ``use_trs_nodes``).

    Returns:
    - bytes: GLB file content.

    Raises:
    - MeshExportError: the geometry or the skeleton cannot be exported.
    - SkinDataError: the per-vertex influences are unusable.
    """
    scene = build_scene(mesh, **options)

    gltf_data = dict(scene.json_data)
    if scene.binary:
        # A GLB buffer with no uri refers to the BIN chunk.
        gltf_data["buffers"] = [{"byteLength": len(scene.binary)}]

    json_bytes = json.dumps(gltf_data, separators=(",", ":")).encode("utf-8")
    return pack_glb(json_bytes, scene.binary)
