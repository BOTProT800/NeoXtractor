"""
End-to-end verification through Blender's glTF importer.

Blender cannot run inside the project environment: the ``bpy`` wheel is tied to
one CPython version and weighs hundreds of megabytes, so it is not a project
dependency. Instead these tests shell out to a separate interpreter that has
``bpy`` installed, named by the ``NEOX_BLENDER_PYTHON`` environment variable,
and skip when it is absent.

Setting one up, on Windows (``bpy`` 4.5 LTS needs CPython 3.11)::

    uv venv --python 3.11 C:/tmp/blv
    uv pip install --python C:/tmp/blv/Scripts/python.exe bpy==4.5.14
    set NEOX_BLENDER_PYTHON=C:/tmp/blv/Scripts/python.exe

Keep that path short. Blender's addon tree is deeply nested and a long prefix
pushes ``io_scene_gltf2`` past the Windows 260 character path limit, where
Python silently fails to import parts of the glTF addon.

The heavy lifting lives in :mod:`tests.support.blender_check`, which is also
usable directly on a real model::

    <blender-python> tests/support/blender_check.py model.glb
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from core.mesh_converter.formats import glb
from tests.support.synthetic import (
    asymmetric_character,
    bone_255_mesh,
    box_figure,
    multi_root_mesh,
    reversed_storage_mesh,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "tests" / "support" / "blender_check.py"

blender_python = pytest.mark.skipif(
    not os.environ.get("NEOX_BLENDER_PYTHON"),
    reason="set NEOX_BLENDER_PYTHON to an interpreter with bpy installed",
)


def run_blender_check(payload: bytes, tmp_path: Path, name: str) -> str:
    """Write a GLB and run the Blender checker on it, returning its output."""
    interpreter = os.environ["NEOX_BLENDER_PYTHON"]
    target = tmp_path / f"{name}.glb"
    target.write_bytes(payload)

    process = subprocess.run(
        [interpreter, str(CHECKER), str(target)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=900,
    )
    output = process.stdout + process.stderr
    assert "RESULT ok" in output, output[-4000:]
    return output


@blender_python
@pytest.mark.parametrize(
    "name, factory",
    [
        ("character", asymmetric_character),
        ("character16", lambda: asymmetric_character(joint_index_bits=16)),
        ("multiroot", multi_root_mesh),
        ("reversed", reversed_storage_mesh),
        ("bone255", bone_255_mesh),
    ],
)
def test_blender_imports_and_deforms_the_rig(name, factory, tmp_path):
    """
    Blender resolves the skin and poses it the way the maths predicts.

    The checker verifies, inside Blender: rest geometry, that every bone head
    sits at its exported joint origin, that the bones are *not* all collapsed
    onto the origin, the parent hierarchy, the vertex groups and their weights,
    and that an armature-space delta applied to a bone moves exactly the
    vertices of its subtree and nothing else.
    """
    run_blender_check(glb.convert(factory()), tmp_path, name)


@blender_python
def test_blender_imports_and_plays_an_animation(tmp_path):
    """
    A clip survives the round trip and moves the rig inside Blender.

    The checker confirms the action is imported, that sampling it partway
    through actually displaces the mesh, that the displacement ramps rather
    than jumping, and that every animated node is a joint of the skin.
    """
    from core.mesh_converter.animation import rotation_clip
    from core.mesh_converter.gltf_scene import build_scene

    mesh = asymmetric_character()
    skeleton = build_scene(mesh).skeleton
    clip = rotation_clip(
        skeleton,
        bone=1,
        axis="z",
        degrees_over_time=[(0.0, 0.0), (0.5, 60.0), (1.0, 0.0)],
        name="bend",
    )

    run_blender_check(glb.convert(mesh, animations=[clip]), tmp_path, "animated")


RENDERER = REPO_ROOT / "tools" / "render_glb.py"


def colour_mask(cell, channel):
    """Pixels where one channel clearly dominates: the axis arrows and labels."""
    rgb = cell[..., :3]
    others = [index for index in range(3) if index != channel]
    main = rgb[..., channel]
    return (
        (main > 0.35)
        & (main > 1.6 * rgb[..., others[0]])
        & (main > 1.6 * rgb[..., others[1]])
    )


def model_mask(cell):
    """The grey model: neutral, and darker than the backdrop."""
    rgb = cell[..., :3]
    value = rgb.mean(axis=-1)
    return (value > 0.3) & (value < 0.88) & (rgb.max(axis=-1) - rgb.min(axis=-1) < 0.1)


def render_sheet(tmp_path, payload, *arguments):
    """Run tools/render_glb.py on a GLB; return the sheet as RGBA floats."""
    from PIL import Image

    target = tmp_path / "model.glb"
    target.write_bytes(payload)
    picture = tmp_path / "model.png"
    process = subprocess.run(
        [
            os.environ["NEOX_BLENDER_PYTHON"],
            str(RENDERER),
            str(target),
            "--out",
            str(picture),
            *arguments,
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=900,
    )
    assert process.returncode == 0, (process.stdout + process.stderr)[-4000:]
    return np.asarray(Image.open(picture).convert("RGBA"), dtype=np.float64) / 255.0


@blender_python
def test_the_render_shows_each_side_where_its_caption_says(tmp_path):
    """
    A picture is only a check if its captions are true.

    ``tools/render_glb.py`` exists so a person can compare an export with the
    game, which is the check that caught the mirrored export when 188 tests
    did not. This reads the rendered sheet back and confirms, from pixels:

    * seen from +Z the red +X arrow is right of the green +Y one, seen from -Z
      it is left of it, and seen from +X the blue +Z arrow points left;
    * the figure's staff, the only thing above its head and held in its left
      hand at +X, is right of the axis seen from the front and left of it
      seen from the back;
    * the clip row shows the rig actually moving.

    It does not, and cannot, say NeoX needs no mirror; see ``NEOX_TO_GLTF``.
    """
    from core.mesh_converter.animation import rotation_clip
    from core.mesh_converter.gltf_scene import build_scene

    mesh = box_figure()
    skeleton = build_scene(mesh).skeleton
    wave = rotation_clip(
        skeleton,
        bone=4,  # upperarm_l, the staff arm
        axis="z",
        degrees_over_time=[(0.0, 0.0), (0.5, 70.0), (1.0, 0.0)],
        name="wave_left",
    )
    size = 240
    sheet = render_sheet(
        tmp_path,
        glb.convert(mesh, animations=[wave]),
        "--clip",
        "wave_left",
        "--frames",
        "3",
        "--size",
        str(size),
    )
    assert sheet.shape[:2] == (3 * size, 4 * size), "solid, skeleton and clip rows"

    def cell(row, column):
        return sheet[row * size : (row + 1) * size, column * size : (column + 1) * size]

    def centre_x(mask):
        columns = np.nonzero(mask)[1]
        assert len(columns), "expected pixels of this colour in the view"
        return columns.mean()

    red, green, blue = 0, 1, 2
    front, side, back, top = (cell(0, column) for column in range(4))

    assert centre_x(colour_mask(front, red)) > centre_x(colour_mask(front, green))
    assert centre_x(colour_mask(back, red)) < centre_x(colour_mask(back, green))
    assert centre_x(colour_mask(side, blue)) < centre_x(colour_mask(side, green))
    assert centre_x(colour_mask(top, red)) > centre_x(colour_mask(top, green))

    def staff_x(view):
        """Centre of the model's topmost rows, below the caption band."""
        mask = model_mask(view)
        mask[: size // 10] = False
        rows = np.nonzero(mask.any(axis=1))[0]
        top_row, bottom_row = rows.min(), rows.max()
        band = mask.copy()
        band[top_row + (bottom_row - top_row) // 10 :] = False
        return centre_x(band)

    assert staff_x(front) > centre_x(colour_mask(front, green)), (
        "seen from +Z the staff in the left hand must be on the right"
    )
    assert staff_x(back) < centre_x(colour_mask(back, green))

    rest_frame, raised_frame = cell(2, 0), cell(2, 1)
    moved = np.abs(rest_frame[..., :3] - raised_frame[..., :3]).max(axis=-1) > 0.2
    assert moved.mean() > 0.005, "the clip row should show the arm moving"


@blender_python
def test_a_winding_against_the_normals_renders_dark(tmp_path):
    """
    A winding that disagrees with the stored normals cannot pass for healthy.

    Blender turns a face's normal around when it sees the face from behind,
    so with the triangles reversed every visible face is lit from the wrong
    side and the figure comes out nearly black. This is the symptom the old
    mirroring export would have shown had it not reversed the winding too.
    """
    size = 160
    figure = box_figure()
    healthy = render_sheet(tmp_path, glb.convert(figure), "--size", str(size))

    figure.mesh.face = [(a, c, b) for a, b, c in figure.mesh.face]
    inside_out = render_sheet(tmp_path, glb.convert(figure), "--size", str(size))

    def body_brightness(sheet):
        """Mean value of the figure's torso in the front view."""
        front = sheet[:size, :size, :3]
        centre = front[int(size * 0.35) : int(size * 0.5), int(size * 0.4) : int(size * 0.55)]
        return centre.mean()

    assert body_brightness(healthy) > 0.35
    assert body_brightness(inside_out) < 0.15
