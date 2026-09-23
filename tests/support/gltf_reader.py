"""
A reader for glTF and GLB files, written against the specification.

It deliberately shares no code and no constants with the writer: the container
fields, the chunk types, the accessor component sizes and the column-major
matrix layout are all re-derived here from the glTF 2.0 specification. If the
writer and this reader agree, they agree about the file format rather than
about a shared helper.
"""

from __future__ import annotations

import base64
import json
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Component type -> (struct format character, byte size, numpy dtype).
_COMPONENT_TYPES = {
    5120: ("b", 1, np.int8),
    5121: ("B", 1, np.uint8),
    5122: ("h", 2, np.int16),
    5123: ("H", 2, np.uint16),
    5125: ("I", 4, np.uint32),
    5126: ("f", 4, np.float32),
}

# Accessor type -> number of components.
_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


class GLTFReadError(ValueError):
    """The file does not parse as glTF or GLB."""


@dataclass
class ParsedGLTF:
    """A parsed glTF document plus its buffers."""

    json_data: dict[str, Any]
    buffers: list[bytes] = field(default_factory=list)
    #: True when the file came in as a GLB container.
    is_binary: bool = False

    def accessor(self, index: int) -> np.ndarray:
        """
        Decode accessor ``index`` into an array.

        Returns:
        - ``(count,)`` for SCALAR, ``(count, components)`` otherwise. MAT4
          accessors come back as ``(count, 4, 4)`` already un-transposed from
          the column-major storage glTF mandates.
        """
        accessors = self.json_data.get("accessors", [])
        if index >= len(accessors):
            raise GLTFReadError(f"accessor {index} does not exist")
        accessor = accessors[index]

        component_type = accessor["componentType"]
        if component_type not in _COMPONENT_TYPES:
            raise GLTFReadError(f"unknown componentType {component_type}")
        fmt, component_size, dtype = _COMPONENT_TYPES[component_type]

        type_str = accessor["type"]
        if type_str not in _TYPE_COMPONENTS:
            raise GLTFReadError(f"unknown accessor type {type_str}")
        components = _TYPE_COMPONENTS[type_str]
        count = accessor["count"]

        if "bufferView" not in accessor:
            values = np.zeros((count, components), dtype=dtype)
        else:
            view = self.json_data["bufferViews"][accessor["bufferView"]]
            buffer = self.buffers[view.get("buffer", 0)]
            base = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
            element_size = component_size * components
            stride = view.get("byteStride") or element_size

            values = np.empty((count, components), dtype=dtype)
            for element in range(count):
                start = base + element * stride
                chunk = buffer[start : start + element_size]
                if len(chunk) != element_size:
                    raise GLTFReadError(
                        f"accessor {index} element {element} runs past the buffer"
                    )
                values[element] = struct.unpack(f"<{components}{fmt}", chunk)

        if type_str == "MAT4":
            # glTF stores matrices column-major; transpose each one back into
            # the column-vector form used for maths.
            return np.array(
                [matrix.reshape(4, 4).T for matrix in values.astype(np.float64)]
            )
        if type_str == "SCALAR":
            return values.reshape(-1)
        return values


def _decode_uri(uri: str) -> bytes:
    prefix = "base64,"
    if not uri.startswith("data:"):
        raise GLTFReadError("only data URIs are supported by this reader")
    marker = uri.find(prefix)
    if marker == -1:
        raise GLTFReadError("data URI is not base64 encoded")
    return base64.b64decode(uri[marker + len(prefix) :])


def read_gltf(payload: bytes) -> ParsedGLTF:
    """
    Parse a ``.gltf`` document with embedded data URIs.

    Returns:
    - The parsed document.
    """
    json_data = json.loads(payload.decode("utf-8"))
    buffers = []
    for buffer in json_data.get("buffers", []):
        uri = buffer.get("uri")
        if uri is None:
            raise GLTFReadError("a .gltf buffer has no uri")
        data = _decode_uri(uri)
        if len(data) < buffer["byteLength"]:
            raise GLTFReadError(
                f"buffer declares {buffer['byteLength']} bytes but decodes to {len(data)}"
            )
        buffers.append(data)
    return ParsedGLTF(json_data=json_data, buffers=buffers, is_binary=False)


def read_glb(payload: bytes) -> ParsedGLTF:
    """
    Parse a ``.glb`` container from first principles.

    Every structural rule is checked here: the magic, the version, the declared
    total length, the chunk order and the 4 byte alignment of each chunk.

    Returns:
    - The parsed document, with the BIN chunk as buffer 0.
    """
    if len(payload) < 12:
        raise GLTFReadError("file is shorter than a GLB header")
    magic, version, total_length = struct.unpack_from("<III", payload, 0)
    if magic != 0x46546C67:
        raise GLTFReadError(f"bad GLB magic {magic:#x}, expected 'glTF'")
    if version != 2:
        raise GLTFReadError(f"unsupported GLB version {version}")
    if total_length != len(payload):
        raise GLTFReadError(
            f"header declares {total_length} bytes but the file holds {len(payload)}"
        )

    offset = 12
    chunks: list[tuple[int, bytes]] = []
    while offset < total_length:
        if offset + 8 > total_length:
            raise GLTFReadError("truncated chunk header")
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        if chunk_length % 4 != 0:
            raise GLTFReadError(
                f"chunk at {offset - 8} has length {chunk_length}, not a multiple of 4"
            )
        if offset + chunk_length > total_length:
            raise GLTFReadError("chunk runs past the declared file length")
        chunks.append((chunk_type, payload[offset : offset + chunk_length]))
        offset += chunk_length

    if not chunks or chunks[0][0] != 0x4E4F534A:
        raise GLTFReadError("the first chunk must be of type JSON")

    json_data = json.loads(chunks[0][1].decode("utf-8").rstrip(" "))

    binary = b""
    for chunk_type, chunk_data in chunks[1:]:
        if chunk_type == 0x004E4942:
            binary = chunk_data
            break

    buffers: list[bytes] = []
    for index, buffer in enumerate(json_data.get("buffers", [])):
        uri = buffer.get("uri")
        if uri is None:
            if index != 0:
                raise GLTFReadError("only buffer 0 may refer to the BIN chunk")
            if len(binary) < buffer["byteLength"]:
                raise GLTFReadError(
                    f"buffer declares {buffer['byteLength']} bytes but the BIN "
                    f"chunk holds {len(binary)}"
                )
            buffers.append(binary)
        else:
            buffers.append(_decode_uri(uri))

    return ParsedGLTF(json_data=json_data, buffers=buffers, is_binary=True)


def read_any(payload: bytes) -> ParsedGLTF:
    """Parse either container, picked by the GLB magic."""
    if payload[:4] == b"glTF":
        return read_glb(payload)
    return read_gltf(payload)
