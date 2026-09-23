"""PMX Format Converter"""

import io
from typing import cast

import pymeshio.pmx.writer
from pymeshio import common, pmx

from core.logger import get_logger
from core.mesh_converter.skeleton import (
    IDENTITY_CONVERSION,
    SkeletonError,
    build_skeleton,
)
from core.mesh_loader import MeshData

NAME = "Polygon Model eXtended (PMX) Format"
EXTENSION = ".pmx"

# Geometry is written in the source basis, so the skeleton is resolved in that
# same basis. Mesh and bones have to agree.
CONVERSION = IDENTITY_CONVERSION

#: Influence slots a PMX Bdef4 vertex holds.
MAX_LINKS = 4


def convert(mesh: MeshData) -> bytes:
    """
    Convert mesh to PMX format.

    PMX stores each bone's rest position **absolutely**, in model space. The
    previous version read it from ``matrix[0:3, 3]``, the wrong slot for the
    row-vector layout NeoX files use, so every bone landed on the origin; it
    also walked the hierarchy from a single root, dropping the other branches
    when a file had more than one.

    Parameters:
    - mesh: MeshData object containing bones, vertices, faces, etc.

    Returns:
    - bytes: PMX file content as bytes
    """
    pmx_model = pmx.Model()
    pmx_model.display_slots.append(pmx.DisplaySlot("Expression", "Exp", 1, None))
    pmx_model.name = "ExportedMesh_NeoXtractor"
    pmx_model.comment = "Created by NeoXtractor"

    # Build bone hierarchy if bones exist
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
                "PMX: skeleton could not be resolved (%s); "
                "falling back to a single root bone",
                error,
            )

    if skeleton is not None:
        bone_pool: list[pmx.Bone] = []
        # PMX requires a bone's parent to already be defined, so bones are
        # emitted in the skeleton's topological order. That covers several
        # roots and parents stored after their children in one go.
        old2new = {source: position for position, source in enumerate(skeleton.order)}

        for source_index in skeleton.order:
            bone = skeleton.bones[source_index]
            x, y, z = bone.global_rest[:3, 3]
            bone_pool.append(
                pmx.Bone(
                    name=bone.name,
                    english_name=bone.name,
                    position=common.Vector3(float(x), float(y), float(z)),
                    parent_index=-1 if bone.parent == -1 else old2new[bone.parent],
                    layer=0,
                    flag=0,
                )
            )
            bone_pool[-1].setFlag(pmx.BONEFLAG_CAN_ROTATE, True)
            bone_pool[-1].setFlag(pmx.BONEFLAG_IS_VISIBLE, True)
            bone_pool[-1].setFlag(pmx.BONEFLAG_CAN_MANIPULATE, True)

        pmx_model.bones = bone_pool
    else:
        # Create a default root bone
        root_bone = pmx.Bone(
            name="root",
            english_name="root",
            position=common.Vector3(0, 0, 0),
            parent_index=-1,
            layer=0,
            flag=0,
        )
        root_bone.setFlag(pmx.BONEFLAG_CAN_ROTATE, True)
        root_bone.setFlag(pmx.BONEFLAG_IS_VISIBLE, True)
        root_bone.setFlag(pmx.BONEFLAG_CAN_MANIPULATE, True)
        pmx_model.bones = [root_bone]
        old2new = {0: 0}

    sentinel = mesh.bones.joint_index_sentinel

    def influences(vertex_index: int) -> list[tuple[int, float]]:
        """
        Collect a vertex's usable influences as PMX bone indices, heaviest first.

        A slot holding this file's empty-slot sentinel, or an index no bone
        answers to, is skipped. When it carries a positive weight that is a
        data problem, so it is reported instead of being quietly folded onto
        the root the way the previous version did.
        """
        if skeleton is None or vertex_index >= len(mesh.bones.joints):
            return []

        merged: dict[int, float] = {}
        for joint_index, weight in zip(
            mesh.bones.joints[vertex_index], mesh.bones.weights[vertex_index]
        ):
            weight = float(weight)
            if weight <= 0.0:
                continue
            joint_index = int(joint_index)
            if joint_index == sentinel or joint_index not in old2new:
                get_logger().warning(
                    "PMX: vertex %d puts weight %g on bone index %d, which is not a "
                    "usable bone; the influence is dropped",
                    vertex_index,
                    weight,
                    joint_index,
                )
                continue
            target = old2new[joint_index]
            merged[target] = merged.get(target, 0.0) + weight

        ranked = sorted(merged.items(), key=lambda item: -item[1])[:MAX_LINKS]
        total = sum(weight for _, weight in ranked)
        if total <= 0.0:
            return []
        return [(index, weight / total) for index, weight in ranked]

    unweighted = 0
    for i, position in enumerate(mesh.mesh.position):
        x, y, z = position
        nx, ny, nz = mesh.mesh.normal[i]
        u, v = mesh.mesh.uv[i]

        links = influences(i)
        if skeleton is not None and not links:
            unweighted += 1

        if links:
            # Pad to the four slots Bdef4 always carries. Empty slots point at
            # bone 0 with weight 0, which binds nothing.
            padded = links + [(0, 0.0)] * (MAX_LINKS - len(links))
            vertex_joint_index = [index for index, _ in padded]
            vertex_weights = [weight for _, weight in padded]

            vertex = pmx.Vertex(
                common.Vector3(cast(int, x), cast(int, y), cast(int, z)),
                common.Vector3(cast(int, nx), cast(int, ny), cast(int, nz)),
                common.Vector2(cast(int, u), cast(int, v)),
                pmx.Bdef4(*vertex_joint_index, *vertex_weights),
                0.0,
            )
        else:
            # Either the mesh has no rig at all, or this vertex has nothing
            # usable to bind to. PMX gives every vertex a bone, so there is no
            # way to express "unbound"; the count is reported below so the
            # fallback is visible rather than silent.
            vertex = pmx.Vertex(
                common.Vector3(cast(int, x), cast(int, y), cast(int, z)),
                common.Vector3(cast(int, nx), cast(int, ny), cast(int, nz)),
                common.Vector2(cast(int, u), cast(int, v)),
                pmx.Bdef1(0),
                0.0,
            )
        pmx_model.vertices.append(vertex)

    if unweighted:
        get_logger().warning(
            "PMX: %d of %d vertices had no usable influence and were bound to the "
            "first bone, because the format requires one. They will not deform "
            "with the rig as intended.",
            unweighted,
            len(mesh.mesh.position),
        )

    # Add faces
    for face in mesh.mesh.face:
        pmx_model.indices.extend(face)

    # Default single material
    material = pmx.Material(
        name="Material",
        english_name="Material",
        diffuse_color=common.RGB(1, 1, 1),
        alpha=1.0,
        specular_factor=1,
        specular_color=common.RGB(1, 1, 1),
        ambient_color=common.RGB(0, 0, 0),
        flag=0,
        edge_color=common.RGBA(0, 0, 0, 1),
        edge_size=0,
        texture_index=-1,
        sphere_texture_index=-1,
        sphere_mode=pmx.MATERIALSPHERE_NONE,
        toon_sharing_flag=1,
        toon_texture_index=0,
        comment="Auto-Generated Material",
        vertex_count=len(mesh.mesh.face) * 3,
    )
    pmx_model.materials.append(material)

    # Write to bytes buffer
    buffer = io.BytesIO()
    pymeshio.pmx.writer.write(buffer, pmx_model, 1)
    return buffer.getvalue()
