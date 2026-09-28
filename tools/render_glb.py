"""
Render an exported GLB into one picture a person can check against the game.

Why this exists: every test in this project compares the export against what
the code believes the source means. When that belief is wrong they all agree
with each other and pass. That is how a mirrored export went through 188 green
tests, and why the one check that does not share the assumption is to look at
the model and compare it with the game. This script makes the looking one
command, on any machine with Blender as a Python module.

Run it with the interpreter that has ``bpy`` (see ``tests/test_blender_import.py``
for setting one up)::

    <blender-python> tools/render_glb.py modelo.glb
    <blender-python> tools/render_glb.py modelo.glb --clip walk_f --frames 6

It writes ``modelo.png`` next to the input, a contact sheet of:

* **solid**: the model seen from +Z (the glTF front), +X, -Z and +Y;
* **skeleton**: the same views with the mesh faded and every joint drawn and
  linked to its parent, in front of the mesh;
* **clip**, with ``--clip``: frames spread over that clip, seen three-quarter.

Every view carries the glTF axes, labelled: red +X, green +Y, blue +Z. The
cameras are orthographic, so nothing is distorted by perspective. Back faces
are culled, as game engines do. A winding that disagrees with the file's own
normals comes out black: Blender turns a face's normal around when it is seen
from behind, so every visible face is lit from the wrong side. (That happens
with or without culling. A model whose winding *and* normals are both inside
out still looks healthy, at least on boxes seen square-on.)

What the picture cannot tell you is whether it matches the game. A mirrored
export looks perfectly healthy here; it is only wrong next to the original.
Pick something asymmetric (the hand holding a weapon, a logo, a hairstyle)
and compare.
"""

import argparse
import os
import struct
import sys
import tempfile
import zlib

try:
    import bpy  # first: bmesh and mathutils only exist once bpy is loaded
    import bmesh
    import numpy as np
    from mathutils import Matrix, Vector
except ImportError:  # pragma: no cover - depends on the interpreter
    sys.exit(
        "run this with an interpreter that has bpy installed; "
        "tests/test_blender_import.py says how to set one up"
    )

BACKGROUND = (0.94, 0.94, 0.94)
MODEL_COLOR = (0.62, 0.64, 0.68, 1.0)
BONE_COLOR = (1.0, 0.5, 0.05, 1.0)
CAPTION_COLOR = (0.1, 0.1, 0.1, 1.0)

#: (label, direction in glTF axes, colour).
GLTF_AXES = (
    ("+X", (1.0, 0.0, 0.0), (0.85, 0.08, 0.06, 1.0)),
    ("+Y", (0.0, 1.0, 0.0), (0.08, 0.6, 0.12, 1.0)),
    ("+Z", (0.0, 0.0, 1.0), (0.08, 0.25, 0.9, 1.0)),
)

#: (caption, where the camera sits seen from the model, what is up in the
#: picture), in glTF axes.
VIEWS = (
    ("front, camera on +Z", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    ("side, camera on +X", (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ("back, camera on -Z", (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    ("top, camera on +Y", (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
)
THREE_QUARTER = ((0.65, 0.35, 1.0), (0.0, 1.0, 0.0))

#: Opacity of the mesh behind the skeleton.
GHOST_ALPHA = 0.35


def to_blender(vector) -> Vector:
    """glTF (Y up) to Blender (Z up): the rotation the importer applies."""
    x, y, z = vector
    return Vector((x, -z, y))


# --- scene -------------------------------------------------------------------


def load(path: str):
    """Import the file into an empty scene; return its meshes and armatures."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path)
    scene = bpy.context.scene

    in_scene = set(scene.objects)
    meshes, armatures = [], []
    for obj in bpy.data.objects:
        # The importer keeps scratch shapes (an icosphere for bone display) in
        # a collection of its own; they are not part of the model.
        scratch = obj not in in_scene or all(
            collection.name == "glTF_not_exported" for collection in obj.users_collection
        )
        if scratch:
            obj.hide_render = True
        elif obj.type == "MESH":
            meshes.append(obj)
        elif obj.type == "ARMATURE":
            armatures.append(obj)
    return meshes, armatures


def configure(size: int) -> None:
    """Workbench with a studio light, per-object colours and back faces culled."""
    scene = bpy.context.scene
    render = scene.render
    render.engine = "BLENDER_WORKBENCH"
    render.resolution_x = render.resolution_y = size
    render.resolution_percentage = 100
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"

    world = bpy.data.worlds.new("backdrop")
    world.color = BACKGROUND
    scene.world = world

    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = "OBJECT"
    shading.show_backface_culling = True
    shading.show_xray = False
    # Flat faces seen square-on all shade alike; cavity marks their edges.
    shading.show_cavity = True
    shading.cavity_type = "BOTH"


def rest_pose() -> None:
    """Drop every imported action and put the rig back at its bind pose."""
    for obj in bpy.data.objects:
        if obj.animation_data:
            obj.animation_data_clear()
        if obj.type == "ARMATURE":
            for pose_bone in obj.pose.bones:
                pose_bone.matrix_basis = Matrix()
    bpy.context.view_layer.update()


def find_action(name: str):
    """The imported action for a clip, by exact name first, then by prefix."""
    actions = list(bpy.data.actions)
    exact = [action for action in actions if action.name == name]
    if exact:
        return exact[0]
    prefixed = [action for action in actions if action.name.startswith(name)]
    if prefixed:
        return prefixed[0]
    known = ", ".join(sorted(action.name for action in actions)) or "none"
    raise SystemExit(f"no clip named {name!r} in this file; clips: {known}")


def play(armature, action) -> tuple[float, float]:
    """Make ``action`` the only thing driving the armature; return its frames."""
    data = armature.animation_data or armature.animation_data_create()
    data.action = action
    # Blender 4.4+ binds an action through a slot.
    if hasattr(data, "action_slot") and data.action_slot is None and len(action.slots):
        data.action_slot = action.slots[0]
    first, last = action.frame_range
    return float(first), float(last)


def mesh_points(meshes) -> np.ndarray:
    """World-space vertices after the armature deforms them."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    chunks = []
    for obj in meshes:
        evaluated = obj.evaluated_get(depsgraph)
        data = evaluated.to_mesh()
        coordinates = np.empty(len(data.vertices) * 3)
        data.vertices.foreach_get("co", coordinates)
        coordinates = coordinates.reshape(-1, 3)
        matrix = np.array(evaluated.matrix_world)
        chunks.append(coordinates @ matrix[:3, :3].T + matrix[:3, 3])
        evaluated.to_mesh_clear()
    return np.concatenate(chunks) if chunks else np.zeros((0, 3))


# --- markers -----------------------------------------------------------------


def new_object(name: str, bm, color) -> bpy.types.Object:
    """Turn a bmesh into a coloured object linked to the scene."""
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    data = bpy.data.meshes.new(name)
    bm.to_mesh(data)
    bm.free()
    obj = bpy.data.objects.new(name, data)
    obj.color = color
    bpy.context.scene.collection.objects.link(obj)
    return obj


def aligned(start: Vector, direction: Vector) -> Matrix:
    """A frame at ``start`` whose local Z runs along ``direction``."""
    rotation = direction.normalized().to_track_quat("Z", "Y").to_matrix().to_4x4()
    return Matrix.Translation(start) @ rotation


def add_cone(bm, start: Vector, direction: Vector, length, radius_start, radius_end):
    """A closed cone or cylinder from ``start`` along ``direction``."""
    frame = aligned(start, direction) @ Matrix.Translation((0.0, 0.0, length / 2.0))
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        segments=12,
        radius1=radius_start,
        radius2=radius_end,
        depth=length,
        matrix=frame,
    )


def axes_gizmo(scale: float, camera) -> list:
    """The three glTF axes as labelled arrows at the origin."""
    objects = []
    for label, direction, color in GLTF_AXES:
        axis = to_blender(direction)
        bm = bmesh.new()
        add_cone(bm, Vector(), axis, scale * 0.8, scale * 0.025, scale * 0.025)
        add_cone(bm, axis * scale * 0.8, axis, scale * 0.2, scale * 0.07, 0.0)
        objects.append(new_object(f"axis {label}", bm, color))

        text = text_object(f"label {label}", label, scale * 0.22, color)
        text.location = axis * scale * 1.18
        # Keep the label facing whichever view is being rendered.
        constraint = text.constraints.new("COPY_ROTATION")
        constraint.target = camera
        objects.append(text)
    return objects


def text_object(name: str, body: str, size: float, color, align="CENTER"):
    """Flat text, rendered as geometry, so it needs no font overlay."""
    curve = bpy.data.curves.new(name, type="FONT")
    curve.body = body
    curve.size = size
    curve.align_x = align
    curve.align_y = "CENTER" if align == "CENTER" else "TOP"
    obj = bpy.data.objects.new(name, curve)
    obj.color = color
    bpy.context.scene.collection.objects.link(obj)
    return obj


def skeleton_object(armatures, extent: float):
    """Every joint as a ball, linked to its parent by a tapered spike."""
    bm = bmesh.new()
    joint_radius = extent * 0.012
    for armature in armatures:
        world = armature.matrix_world
        for pose_bone in armature.pose.bones:
            head = world @ pose_bone.head
            bmesh.ops.create_uvsphere(
                bm,
                u_segments=10,
                v_segments=6,
                radius=joint_radius,
                matrix=Matrix.Translation(head),
            )
            if pose_bone.parent is None:
                continue
            start = world @ pose_bone.parent.head
            link = head - start
            if link.length < 1e-9:
                continue
            add_cone(bm, start, link, link.length, joint_radius * 0.9, joint_radius * 0.25)
    return new_object("skeleton", bm, BONE_COLOR)


# --- cameras -----------------------------------------------------------------


def frame(camera, caption_object, direction, up, points: np.ndarray, caption: str):
    """Aim the orthographic camera along ``-direction`` so ``points`` fit."""
    back = to_blender(direction).normalized()
    upward = to_blender(up)
    upward = (upward - back * upward.dot(back)).normalized()
    right = upward.cross(back)

    basis = np.array([right, upward, back])
    projected = points @ basis.T
    low, high = projected.min(axis=0), projected.max(axis=0)
    width, height = high[:2] - low[:2]
    depth = float(high[2] - low[2])
    span = max(float(width), float(height), 1e-6)

    centre = right * float((low[0] + high[0]) / 2.0) + upward * float((low[1] + high[1]) / 2.0)
    camera.location = centre + back * (float(high[2]) + span)
    camera.rotation_euler = Matrix((right, upward, back)).transposed().to_euler()
    lens = camera.data
    lens.type = "ORTHO"
    lens.ortho_scale = span * 1.25
    lens.clip_start = span * 1e-3
    lens.clip_end = depth + span * 3.0

    # The caption hangs just in front of the lens, in its top-left corner.
    half = lens.ortho_scale / 2.0
    caption_object.data.body = caption
    caption_object.data.size = lens.ortho_scale * 0.045
    caption_object.location = (-half * 0.95, half * 0.95, -lens.clip_start * 4.0)


# --- images ------------------------------------------------------------------


def shoot(path: str) -> np.ndarray:
    """Render the scene; return straight-alpha RGBA rows, top row first."""
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(path)
    width, height = image.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    # Blender stores rows bottom-up.
    return pixels.reshape(height, width, 4)[::-1]


def over(front: np.ndarray, back: np.ndarray) -> np.ndarray:
    """Composite a transparent render over an opaque one."""
    alpha = front[..., 3:4]
    result = back.copy()
    result[..., :3] = front[..., :3] * alpha + back[..., :3] * (1.0 - alpha)
    result[..., 3] = 1.0
    return result


def write_png(path: str, rows: np.ndarray) -> None:
    """A minimal RGBA PNG writer, so the sheet needs nothing outside bpy."""
    height, width, _ = rows.shape
    data = (np.clip(rows, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    raw = b"".join(b"\x00" + data[row].tobytes() for row in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    with open(path, "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)))
        handle.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        handle.write(chunk(b"IEND", b""))


def sheet(rows: list[list[np.ndarray]], size: int) -> np.ndarray:
    """Tile the rows of pictures, padding short rows with the background."""
    columns = max(len(row) for row in rows)
    canvas = np.ones((size * len(rows), size * columns, 4), dtype=np.float32)
    canvas[..., :3] = BACKGROUND
    for r, row in enumerate(rows):
        for c, picture in enumerate(row):
            canvas[r * size : (r + 1) * size, c * size : (c + 1) * size] = picture
    # A thin rule between cells keeps neighbouring views apart.
    canvas[::size, :, :3] = 0.75
    canvas[:, ::size, :3] = 0.75
    return canvas


# --- main --------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("glb", help="the exported .glb or .gltf")
    parser.add_argument("--out", help="where to write the picture (default: next to the input)")
    parser.add_argument("--clip", help="also show frames of this clip")
    parser.add_argument("--frames", type=int, default=5, help="frames of the clip, ends included (default 5)")
    parser.add_argument("--size", type=int, default=480, help="pixels per view (default 480)")
    args = parser.parse_args(argv)

    source = os.path.abspath(args.glb)
    destination = args.out or os.path.splitext(source)[0] + ".png"

    meshes, armatures = load(source)
    if not meshes:
        print(f"no mesh in {source}")
        return 1
    configure(args.size)
    rest_pose()

    scene = bpy.context.scene
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    caption = text_object("caption", "", 1.0, CAPTION_COLOR, align="LEFT")
    caption.parent = camera

    for obj in meshes:
        obj.color = MODEL_COLOR

    rest = mesh_points(meshes)
    low, high = rest.min(axis=0), rest.max(axis=0)
    extent = float(np.max(high - low)) or 1.0
    gizmo = axes_gizmo(extent * 0.3, camera)
    tips = np.array([to_blender(axis) * extent * 0.3 * 1.3 for _, axis, _ in GLTF_AXES])
    framing = np.concatenate([rest, tips, np.zeros((1, 3))])

    name = os.path.basename(source)
    overlay = [caption, *gizmo]
    rows = []
    with tempfile.TemporaryDirectory() as scratch:
        shots = (os.path.join(scratch, f"{index}.png") for index in range(10_000))

        def render_view(label, direction, up, points, with_skeleton):
            """
            The model underneath, faded when the skeleton is shown, and the
            axes, caption and skeleton on top, the way Blender draws an
            armature set to "In Front". Nothing in the overlay can be hidden
            by the model it describes.
            """
            frame(camera, caption, direction, up, points, f"{name}  {label}")
            extras = list(overlay)
            if with_skeleton:
                extras.append(skeleton_object(armatures, extent))

            shading = scene.display.shading
            for obj in extras:
                obj.hide_render = True
            shading.show_xray = with_skeleton
            shading.xray_alpha = GHOST_ALPHA
            model = shoot(next(shots))
            shading.show_xray = False

            for obj in meshes:
                obj.hide_render = True
            for obj in extras:
                obj.hide_render = False
            scene.render.film_transparent = True
            on_top = shoot(next(shots))
            scene.render.film_transparent = False
            for obj in meshes:
                obj.hide_render = False

            if with_skeleton:
                bpy.data.objects.remove(extras[-1])
            return over(on_top, model)

        rows.append([render_view(*view, framing, False) for view in VIEWS])
        if armatures:
            rows.append([render_view(*view, framing, True) for view in VIEWS])

        if args.clip:
            if not armatures:
                print("the file has no skeleton, so it has no clips to show")
                return 1
            action = find_action(args.clip)
            first, last = play(armatures[0], action)
            count = max(1, args.frames)
            frames = [
                first + (last - first) * index / max(1, count - 1) for index in range(count)
            ]

            def show(number):
                scene.frame_set(int(number), subframe=number - int(number))

            # One camera for every frame, so re-framing cannot hide the motion.
            sampled = []
            for number in frames:
                show(number)
                sampled.append(mesh_points(meshes))
            clip_framing = np.concatenate([*sampled, tips, np.zeros((1, 3))])
            pictures = []
            for number in frames:
                show(number)
                label = f"{action.name}  frame {number:g} of {first:g}-{last:g}"
                pictures.append(render_view(label, *THREE_QUARTER, clip_framing, True))
            rows.append(pictures)
            print(f"clip      {action.name}, frames {', '.join(f'{f:g}' for f in frames)}")

    write_png(destination, sheet(rows, args.size))
    gltf_low = (low[0], low[2], -high[1])
    gltf_high = (high[0], high[2], -low[1])
    print(f"model     {len(rest)} vertices, {sum(len(a.pose.bones) for a in armatures)} bones")
    print(
        "bounds    glTF x {:.3g}..{:.3g}  y {:.3g}..{:.3g}  z {:.3g}..{:.3g}".format(
            gltf_low[0], gltf_high[0], gltf_low[1], gltf_high[1], gltf_low[2], gltf_high[2]
        )
    )
    print(f"wrote     {destination}")
    return 0


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    raise SystemExit(main(arguments))
