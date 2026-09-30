"""
``tools/diagnose_mesh.py`` end to end, on a .mesh and a .gis written to disk.

Its ``--conversion`` option once offered ``neox_flip_x``, which had quietly
become the identity while the default did not mirror: a label that no longer
did what it said, on the one switch meant for comparing a model against its
mirror image. These tests hold each option to its name.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.support.gltf_reader import read_any
from tests.support.synthetic import box_figure, build_mesh_file, build_rgis_file

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "diagnose_mesh.py"
IDENTITY_Q = (0.0, 0.0, 0.0, 1.0)


@pytest.fixture(scope="module")
def figure_files(tmp_path_factory):
    """The box figure as a real .mesh, plus a .gis that waves its left arm."""
    folder = tmp_path_factory.mktemp("diagnose")
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
    # The left arm is at -X in NeoX, so raising it turns it by a negative angle.
    raised = (0.0, 0.0, float(np.sin(np.radians(-35.0))), float(np.cos(np.radians(-35.0))))
    clip = build_rgis_file(
        reference=[("upperarm_l", (-0.35, 0.6, 0.0), IDENTITY_Q, (1.0, 1.0, 1.0))],
        clips=[
            {
                "name": "wave",
                "fps": 30,
                "times": [0.0, 0.5, 1.0],
                "tracks": [
                    ("upperarm_l", (-0.35, 0.6, 0.0), [IDENTITY_Q, raised, IDENTITY_Q], (1.0, 1.0, 1.0))
                ],
            }
        ],
    )
    mesh_path = folder / "figure.mesh"
    mesh_path.write_bytes(blob.data)
    gis_path = folder / "figure.gis"
    gis_path.write_bytes(clip)
    return mesh_path, gis_path, blob


def run_tool(*arguments):
    process = subprocess.run(
        [sys.executable, str(TOOL), *map(str, arguments)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=300,
    )
    assert process.returncode == 0, process.stdout[-3000:] + process.stderr[-3000:]
    return process.stdout


def exported_positions(path):
    document = read_any(Path(path).read_bytes())
    primitive = document.json_data["meshes"][0]["primitives"][0]
    return document, document.accessor(primitive["attributes"]["POSITION"])


def test_each_conversion_does_what_its_name_says(figure_files, tmp_path):
    mesh_path, _, blob = figure_files
    source = np.asarray(blob.positions, dtype=np.float64)

    default_out = tmp_path / "default.glb"
    report = run_tool(mesh_path, "--out", default_out)
    assert "coordinate conversion: mirror_x" in report
    _, default = exported_positions(default_out)
    # Half floats in the file: compare at their precision.
    assert np.allclose(default, source * np.array([-1.0, 1.0, 1.0]), atol=2e-3)

    plain_out = tmp_path / "plain.glb"
    report = run_tool(mesh_path, "--out", plain_out, "--conversion", "identity")
    assert "coordinate conversion: identity" in report
    _, plain = exported_positions(plain_out)
    assert np.allclose(plain, default * np.array([-1.0, 1.0, 1.0]), atol=1e-6)


def test_the_clip_rides_along(figure_files, tmp_path):
    mesh_path, gis_path, _ = figure_files
    out = tmp_path / "animated.glb"
    report = run_tool(mesh_path, "--anim", gis_path, "--out", out)

    assert "1 read, 0 skipped" in report
    document, _ = exported_positions(out)
    assert [clip["name"] for clip in document.json_data["animations"]] == ["wave"]
