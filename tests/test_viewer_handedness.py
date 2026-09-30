"""
NeoXtractor's own viewer must show the model the export shows.

NeoX is left-handed and the viewer's camera right-handed, so the viewer
mirrors X, as the glTF export does. For a few days in September 2026 first
the export and then the viewer dropped that mirror; each side agreed with its
own idea of the data and no test noticed. A real rig's left/right bone names
did (see ``NEOX_TO_GLTF``).

The first test pins what the viewer uploads. The second opens the real viewer
widget, takes its picture and reads the pixels. It needs a screen, so it runs
only with ``NEOX_VIEWER_TESTS=1``; on Linux without one, run pytest under
``xvfb-run -a``.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gui.renderers.mesh_renderer import ProcessedMeshData
from gui.widgets.viewers.mesh_viewer.camera import Camera, OrthogonalDirection
from tests.support.synthetic import box_figure, build_mesh_file

REPO_ROOT = Path(__file__).resolve().parents[1]
CAPTURE = REPO_ROOT / "tools" / "capture_viewer.py"


def test_the_viewer_draws_what_the_glb_export_writes():
    """X mirrored, triangles turned back, bones mirrored with the mesh."""
    from core.mesh_converter.formats import glb
    from tests.support.gltf_reader import read_any

    figure = box_figure()
    processed = ProcessedMeshData(figure)

    document = read_any(glb.convert(figure))
    primitive = document.json_data["meshes"][0]["primitives"][0]
    assert np.allclose(processed.vertices[:, :3], document.accessor(primitive["attributes"]["POSITION"]), atol=1e-6)
    assert np.allclose(processed.vertices[:, 3:], document.accessor(primitive["attributes"]["NORMAL"]), atol=1e-6)

    faces = np.asarray(figure.mesh.face)
    assert np.array_equal(processed.indices, faces[:, [1, 0, 2]])

    # The left forearm: -X in NeoX, +X on screen as in the GLB.
    forearm_l = figure.bones.names.index("forearm_l")
    assert np.allclose(processed.bone_positions[forearm_l], (0.9, 2.1, 0.0))


@pytest.mark.parametrize(
    "direction, opposite, expected",
    [
        (OrthogonalDirection.FRONT, False, (0.0, 0.0, 1.0)),
        (OrthogonalDirection.FRONT, True, (0.0, 0.0, -1.0)),
        (OrthogonalDirection.RIGHT, False, (-1.0, 0.0, 0.0)),
        (OrthogonalDirection.RIGHT, True, (1.0, 0.0, 0.0)),
    ],
)
def test_each_view_key_keeps_showing_the_same_side(direction, opposite, expected):
    """
    Where the camera sits for the view keys, on screen (the mirrored space).

    Key 3 puts the camera at the screen's -X, which is the model's +X in
    NeoX coordinates: the side it has always shown.
    """
    from PySide6.QtGui import QVector4D

    camera = Camera()
    camera.orthogonal(direction, opposite)
    toward = camera.rot().inverted()[0].map(QVector4D(0.0, 0.0, 1.0, 0.0))
    assert np.allclose((toward.x(), toward.y(), toward.z()), expected, atol=1e-6)


@pytest.mark.skipif(
    not os.environ.get("NEOX_VIEWER_TESTS"),
    reason="needs a screen: set NEOX_VIEWER_TESTS=1 (under xvfb-run -a on Linux)",
)
def test_the_real_viewer_shows_the_staff_where_the_export_does(tmp_path):
    """
    The figure holds a staff in its left hand.

    Seen from +Z (key 1) it must be on the right of the picture, as it is in
    ``tools/render_glb.py``'s front view of the exported GLB, and on the left
    from -Z (Ctrl+1). Without the viewer's mirror both come out the other way.
    """
    from PIL import Image

    figure = box_figure()
    bones = figure.bones
    blob = build_mesh_file(
        positions=figure.mesh.position,
        normals=figure.mesh.normal,
        faces=figure.mesh.face,
        uvs=figure.mesh.uv,
        joints=bones.joints,
        weights=bones.weights,
        bone_parents=bones.parents,
        bone_names=bones.names,
        bone_matrices=bones.matrix,
    )
    source = tmp_path / "figure.mesh"
    source.write_bytes(blob.data)
    picture = tmp_path / "viewer.png"
    size = 300

    process = subprocess.run(
        [sys.executable, str(CAPTURE), str(source), "--out", str(picture), "--size", str(size)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=300,
    )
    assert process.returncode == 0, process.stdout[-2000:] + process.stderr[-2000:]

    strip = np.asarray(Image.open(picture).convert("RGB"), dtype=np.float64) / 255.0
    assert strip.shape[:2] == (size, 4 * size)

    def staff_x(column):
        """Centre of the model's topmost rows, where only the staff reaches."""
        view = strip[:, column * size : (column + 1) * size]
        value = view.mean(axis=-1)
        neutral = view.max(axis=-1) - view.min(axis=-1) < 0.06
        model = neutral & (value > 0.3)
        model[:30] = False  # the caption
        rows = np.nonzero(model.any(axis=1))[0]
        assert len(rows), "the model should be visible in the view"
        top, bottom = rows.min(), rows.max()
        band = model.copy()
        band[top + (bottom - top) // 10 :] = False
        return np.nonzero(band)[1].mean()

    front, back = 0, 2
    assert staff_x(front) > size / 2, "seen from +Z the staff must be on the right"
    assert staff_x(back) < size / 2, "seen from -Z the staff must be on the left"
