"""Mesh conversion utilities."""

from core.mesh_loader import MeshData

from .formats import ascii as mesh_ascii
from .formats import fbx, glb, gltf, iqe, obj, pmx, smd
from .gltf_scene import GLTFScene, MeshExportError, SkinDataError, build_scene

FORMATS = [mesh_ascii, glb, gltf, iqe, obj, pmx, smd, fbx]


def convert_mesh(mesh: MeshData, target_format: type, **kwargs) -> bytes:
    """
    Convert mesh to the specified format.

    Parameters:
    - mesh: MeshData object containing bones, vertices, faces, etc.
    - target_format: The format class to convert the mesh to.
    - kwargs: Additional parameters for specific format conversions.

    Returns:
    - bytes: Converted mesh data in the specified format.
    """
    return target_format.convert(mesh, **kwargs)


__all__ = [
    "convert_mesh",
    "build_scene",
    "GLTFScene",
    "MeshExportError",
    "SkinDataError",
    "FORMATS",
    "mesh_ascii",
    "glb",
    "gltf",
    "iqe",
    "obj",
    "pmx",
    "smd",
    "fbx",
]
