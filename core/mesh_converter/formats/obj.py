"""Object File Format (OBJ) Converter"""

from core.mesh_loader import MeshData

NAME = "Wavefront (OBJ) Format - No Skeleton"
EXTENSION = ".obj"


def convert(mesh: MeshData, flip_uv=False) -> bytes:
    """
    Convert mesh to OBJ format as a static mesh without skeleton.

    Parameters:
    - mesh: MeshData object to be converted to OBJ format.
    - flip_uv: Boolean to indicate whether to flip the UV coordinates on the Y-axis.

    Returns:
    - bytes: OBJ file content as bytes
    """
    obj_lines = []
    obj_lines.append("o Neox Mesh\n")

    # OBJ is right-handed and NeoX left-handed (see NEOX_TO_GLTF), so X is
    # mirrored, as zhouhang95/neox_tools does, and each triangle's order is
    # reversed below so its winding still agrees with its normals.
    for v in mesh.mesh.position:
        obj_lines.append(f"v {-v[0]} {v[1]} {v[2]}\n")

    # Write normals
    for n in mesh.mesh.normal:
        obj_lines.append(f"vn {-n[0]} {n[1]} {n[2]}\n")

    for uv in mesh.mesh.uv:
        if flip_uv:
            uv = (uv[0], 1 - uv[1])  # Flip UV on the Y axis
        obj_lines.append(f"vt {uv[0]} {uv[1]}\n")

    # Write all faces, two corners swapped to go with the mirror
    for a, b, c in mesh.mesh.face:
        v1, v2, v3 = a, c, b
        if mesh.has_uvs:
            obj_lines.append(
                f"f {v1 + 1}/{v1 + 1} {v2 + 1}/{v2 + 1} {v3 + 1}/{v3 + 1}\n"
            )
        else:
            obj_lines.append(f"f {v1 + 1} {v2 + 1} {v3 + 1}\n")

    return "".join(obj_lines).encode("utf-8")
