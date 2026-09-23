"""ASCII Mesh Format Converter"""

from core.logger import get_logger
from core.mesh_converter.skeleton import (
    IDENTITY_CONVERSION,
    SkeletonError,
    build_skeleton,
    quaternion_from_matrix,
)
from core.mesh_loader import MeshData

NAME = "Text Mesh (ASCII) Format"
EXTENSION = ".ascii"

# Geometry is written in the source basis, so the skeleton is resolved in that
# same basis. Mesh and bones have to agree; which basis it is matters less.
CONVERSION = IDENTITY_CONVERSION


def convert(mesh: MeshData, flip_uv=False) -> bytes:
    """
    Convert mesh to ASCII format.

    Bone lines carry the bone's rest position and orientation in model space,
    taken through :mod:`core.mesh_converter.skeleton` so the matrix layout is
    resolved once for every exporter. The previous version read
    ``matrix.flatten()[:3]``, which is the first row of the stored array and
    not a translation under any convention, and wrote a hardcoded identity
    quaternion.

    Parameters:
    - mesh: MeshData object containing bones, vertices, faces, etc.
    - flip_uv: Boolean to indicate whether to flip the UV coordinates on the Y-axis.

    Returns:
    - bytes: ASCII file content as bytes
    """
    ascii_lines = []

    skeleton = None
    if mesh.has_bones and mesh.bones.names:
        try:
            skeleton = build_skeleton(
                list(mesh.bones.parents),
                list(mesh.bones.names),
                list(mesh.bones.matrix),
                conversion=CONVERSION,
                mesh_positions=mesh.mesh.position,
            )
        except SkeletonError as error:
            get_logger().warning(
                "ASCII: skeleton could not be resolved (%s); "
                "writing bones without transforms",
                error,
            )

    # Write Bone Count
    if mesh.has_bones:
        ascii_lines.append(f"{len(mesh.bones.names)}\n")

        # Write Bone Information
        for i, (name, parent) in enumerate(zip(mesh.bones.names, mesh.bones.parents)):
            ascii_lines.append(f"{name}\n")
            ascii_lines.append(f"{parent}\n")
            if skeleton is not None and i < len(skeleton.bones):
                global_rest = skeleton.bones[i].global_rest
                position = " ".join(f"{value:.6f}" for value in global_rest[:3, 3])
                x, y, z, w = quaternion_from_matrix(global_rest[:3, :3])
                ascii_lines.append(f"{position} {x:.6f} {y:.6f} {z:.6f} {w:.6f}\n")
            else:
                ascii_lines.append("0.000000 0.000000 0.000000 0 0 0 1\n")
    else:
        ascii_lines.append("0\n")

    # Write Vertex Positions
    ascii_lines.append(f"{len(mesh.mesh.position)}\n")
    for x, y, z in mesh.mesh.position:
        ascii_lines.append(f"{x:.6f} {y:.6f} {z:.6f}\n")

    ascii_lines.append(f"{len(mesh.mesh.normal)}\n")
    for nx, ny, nz in mesh.mesh.normal:
        ascii_lines.append(f"{nx:.6f} {ny:.6f} {nz:.6f}\n")

    # Write UVs, applying flip if needed
    if mesh.has_uvs:
        ascii_lines.append(f"{len(mesh.mesh.uv)}\n")
        for u, v in mesh.mesh.uv:
            if flip_uv:
                v = 1 - v
            ascii_lines.append(f"{u:.6f} {v:.6f}\n")
    else:
        ascii_lines.append("0\n")

    # Write Face Indices
    ascii_lines.append(f"{len(mesh.mesh.face)}\n")
    for v1, v2, v3 in mesh.mesh.face:
        ascii_lines.append(f"{v1} {v2} {v3}\n")

    return "".join(ascii_lines).encode("utf-8")
