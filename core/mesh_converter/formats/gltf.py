"""
glTF 2.0 Format Converter (``.gltf``, JSON with an embedded buffer).

The scene itself is built by :mod:`core.mesh_converter.gltf_scene`, which the
``.glb`` writer shares. This module only decides how the binary blob travels:
here it is a base64 data URI, there it is a binary chunk.
"""

import base64
import json

from core.mesh_converter.gltf_scene import (
    MeshExportError,
    SkinDataError,
    build_scene,
)
from core.mesh_loader import MeshData

NAME = "glTF 2.0 (GLTF) Format"
EXTENSION = ".gltf"

__all__ = ["NAME", "EXTENSION", "convert", "MeshExportError", "SkinDataError"]


def convert(mesh: MeshData, **options) -> bytes:
    """
    Convert mesh to glTF format.

    Parameters:
    - mesh: MeshData object containing bones, vertices, faces, etc.
    - options: forwarded to :func:`core.mesh_converter.gltf_scene.build_scene`
      (``name``, ``conversion``, ``matrix_storage``, ``matrix_role``,
      ``use_trs_nodes``).

    Returns:
    - bytes: glTF file content as bytes (JSON with an embedded base64 buffer)

    Raises:
    - MeshExportError: the geometry or the skeleton cannot be exported.
    - SkinDataError: the per-vertex influences are unusable.
    """
    scene = build_scene(mesh, **options)

    gltf_data = dict(scene.json_data)
    if scene.binary:
        encoded = base64.b64encode(scene.binary).decode("ascii")
        gltf_data["buffers"] = [
            {
                "uri": f"data:application/octet-stream;base64,{encoded}",
                "byteLength": len(scene.binary),
            }
        ]

    return json.dumps(gltf_data, separators=(",", ":")).encode("utf-8")
