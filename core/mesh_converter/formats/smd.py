"""Source Model Data (SMD) Format Converter"""

from core.logger import get_logger
from core.mesh_converter.skeleton import (
    IDENTITY_CONVERSION,
    SkeletonError,
    build_skeleton,
    euler_xyz_from_matrix,
)
from core.mesh_loader import MeshData

NAME = "Source Model Data (SMD) Format"
EXTENSION = ".smd"

# Geometry is written in the source basis, so the skeleton is resolved in that
# same basis. Mesh and bones have to agree.
CONVERSION = IDENTITY_CONVERSION

#: Influence slots SMD writes per vertex.
MAX_LINKS = 4


def convert(mesh: MeshData, flip_uv=False) -> bytes:
    """
    Convert mesh to Valve's SMD format as a reference (static) SMD.

    The ``skeleton`` block holds each bone's transform **relative to its
    parent**, as a translation plus XYZ Euler angles in radians. The previous
    version read the translation from ``matrix[0:3, 3]`` -- the wrong slot for
    the row-vector layout NeoX files use, which yields zeros -- treated it as
    if it were parent-relative, and hardcoded the rotation to zero, so every
    bone ended up at its parent's origin with no orientation.

    Parameters:
    - mesh: MeshData object containing bones, vertices, faces, etc.
    - flip_uv: Boolean to indicate whether to flip the UV coordinates on the Y-axis.

    Returns:
    - bytes: SMD file content as bytes
    """
    smd_lines = []
    smd_lines.append("version 1\n")

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
                "SMD: skeleton could not be resolved (%s); "
                "falling back to a single root bone",
                error,
            )

    if skeleton is not None:
        # Nodes are written by ascending index. SMD identifies bones by the id
        # in this block and parent ids may point anywhere in it, so no
        # reordering is needed -- and unlike the previous depth-first walk,
        # this cannot drop a branch when the file has several roots, nor raise
        # when it has none.
        smd_lines.append("nodes\n")
        for bone in skeleton.bones:
            smd_lines.append(f'{bone.source_index} "{bone.name}" {bone.parent}\n')
        smd_lines.append("end\n")

        # Skeleton - Static, only the initial frame at time 0
        smd_lines.append("skeleton\n")
        smd_lines.append("time 0\n")
        for bone in skeleton.bones:
            x, y, z = bone.local_rest[:3, 3]
            rx, ry, rz = euler_xyz_from_matrix(bone.local_rest)
            smd_lines.append(
                f"{bone.source_index} {x:.6f} {y:.6f} {z:.6f} "
                f"{rx:.6f} {ry:.6f} {rz:.6f}\n"
            )
        smd_lines.append("end\n")
    else:
        # No bones - create a single root bone
        smd_lines.append("nodes\n")
        smd_lines.append('0 "root" -1\n')
        smd_lines.append("end\n")

        smd_lines.append("skeleton\n")
        smd_lines.append("time 0\n")
        smd_lines.append("0 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000\n")
        smd_lines.append("end\n")

    bone_count = len(skeleton.bones) if skeleton is not None else 0
    sentinel = mesh.bones.joint_index_sentinel

    def influences(vertex_index: int) -> list[tuple[int, float]]:
        """
        Collect a vertex's usable influences, heaviest first.

        Slots that hold the empty-slot sentinel for this file's index width,
        or an index no bone answers to, are dropped only when they carry no
        weight. A positive weight on such a slot is a data problem, and it is
        reported rather than quietly moved onto the root.
        """
        if skeleton is None or vertex_index >= len(mesh.bones.joints):
            return []

        merged: dict[int, float] = {}
        joints = mesh.bones.joints[vertex_index]
        weights = mesh.bones.weights[vertex_index]
        for joint_index, weight in zip(joints, weights):
            weight = float(weight)
            if weight <= 0.0:
                continue
            joint_index = int(joint_index)
            if joint_index == sentinel or not 0 <= joint_index < bone_count:
                get_logger().warning(
                    "SMD: vertex %d puts weight %g on bone index %d, which is not a "
                    "usable bone; the influence is dropped",
                    vertex_index,
                    weight,
                    joint_index,
                )
                continue
            merged[joint_index] = merged.get(joint_index, 0.0) + weight

        ranked = sorted(merged.items(), key=lambda item: -item[1])[:MAX_LINKS]
        total = sum(weight for _, weight in ranked)
        if total <= 0.0:
            return []
        return [(index, weight / total) for index, weight in ranked]

    # Mesh Data - Vertices, Normals, UVs, and Faces
    smd_lines.append("triangles\n")
    for face_idx, (v1, v2, v3) in enumerate(mesh.mesh.face):
        material_name = f"material_{face_idx // 100}"  # Group faces by material
        smd_lines.append(f"{material_name}\n")

        for vertex_index in [v1, v2, v3]:
            pos = mesh.mesh.position[vertex_index]
            norm = mesh.mesh.normal[vertex_index]
            uv = mesh.mesh.uv[vertex_index]

            # Flip UV if specified
            if flip_uv:
                uv = (uv[0], 1 - uv[1])

            links = influences(vertex_index)
            # The leading id is the vertex's dominant bone; the links that
            # follow carry the full weighting, which the old version dropped
            # entirely.
            bone_id = links[0][0] if links else 0

            line = (
                f"{bone_id} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                f"{norm[0]:.6f} {norm[1]:.6f} {norm[2]:.6f} "
                f"{uv[0]:.6f} {uv[1]:.6f}"
            )
            if links:
                line += f" {len(links)}"
                for index, weight in links:
                    line += f" {index} {weight:.6f}"
            smd_lines.append(line + "\n")

    smd_lines.append("end\n")

    return "".join(smd_lines).encode("utf-8")
