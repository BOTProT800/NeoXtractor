"""
The official Khronos glTF-Validator on every kind of file the exporter writes.

This was the one acceptance condition of the first goal left open: the
validator is not on PyPI, so it is run from its npm package through Node. See
:mod:`tests.support.khronos` for setting it up; without it these tests skip,
the same way the Blender ones do.

``tests/support/gltf_spec_check.py`` reimplements the rules this pipeline can
break and runs everywhere. This is the reference it was standing in for.

The bar is zero errors and zero warnings. Infos are allowed only when listed
in ``EXPECTED_INFOS`` with a reason, so a new one is noticed rather than
scrolled past.
"""

import pytest

from core.anim_loader import read_rgis
from core.mesh_converter.animation import clips_from_rgis, rotation_clip
from core.mesh_converter.formats import glb, gltf
from core.mesh_converter.gltf_scene import build_scene
from core.mesh_converter.skeleton import MIRROR_X
from tests.support import khronos
from tests.support.synthetic import (
    asymmetric_character,
    bone_255_mesh,
    box_figure,
    build_rgis_file,
    make_mesh_data,
    multi_root_mesh,
    reversed_storage_mesh,
)

IDENTITY_Q = (0.0, 0.0, 0.0, 1.0)

pytestmark = pytest.mark.skipif(
    not khronos.available(),
    reason="set NEOX_GLTF_VALIDATOR to the gltf-validator npm package, with node on PATH",
)


def rgis_clips(mesh):
    """A clip read back through the RGIS reader, with every channel animated."""
    turned = (0.0, 0.0, 0.3826834, 0.9238795)
    data = build_rgis_file(
        reference=[
            ("root", (0.0, 0.0, 0.0), IDENTITY_Q, (1.0, 1.0, 1.0)),
            ("arm_l", (1.0, 2.0, 0.0), IDENTITY_Q, (1.0, 1.0, 1.0)),
        ],
        clips=[
            {
                "name": "wave",
                "fps": 30,
                "times": [0.0, 0.5, 1.0],
                "tracks": [
                    (
                        "arm_l",
                        [(1.0, 2.0, 0.0), (1.0, 2.5, 0.0), (1.0, 2.0, 0.0)],
                        [IDENTITY_Q, turned, IDENTITY_Q],
                        [(1.0, 1.0, 1.0), (1.2, 1.0, 1.0), (1.0, 1.0, 1.0)],
                    ),
                    ("root", (0.0, 0.0, 0.0), IDENTITY_Q, (1.0, 1.0, 1.0)),
                ],
            }
        ],
    )
    clips, _ = clips_from_rgis(read_rgis(data), build_scene(mesh).skeleton)
    return clips


def figure_clips(mesh):
    """Linear and step clips on the figure."""
    skeleton = build_scene(mesh).skeleton
    return [
        rotation_clip(
            skeleton, 4, "z", [(0.0, 0.0), (0.5, 70.0), (1.0, 0.0)], name="wave_left"
        ),
        rotation_clip(
            skeleton,
            3,
            "y",
            [(0.0, 0.0), (0.5, 30.0), (1.0, 0.0)],
            name="look",
            interpolation="STEP",
        ),
    ]


def exports():
    """(name, payload) for each distinct shape of output."""
    character = asymmetric_character()
    figure = box_figure()
    static = make_mesh_data(
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], faces=[(0, 1, 2)]
    )
    return [
        ("character.glb", lambda: glb.convert(character)),
        ("character16.glb", lambda: glb.convert(asymmetric_character(16))),
        ("multiroot.glb", lambda: glb.convert(multi_root_mesh())),
        ("reversed.glb", lambda: glb.convert(reversed_storage_mesh())),
        ("bone255.glb", lambda: glb.convert(bone_255_mesh())),
        ("static.glb", lambda: glb.convert(static)),
        ("baked_matrices.glb", lambda: glb.convert(figure, use_trs_nodes=False)),
        ("mirrored.glb", lambda: glb.convert(figure, conversion=MIRROR_X)),
        ("figure_clips.glb", lambda: glb.convert(figure, animations=figure_clips(figure))),
        ("rgis_clip.glb", lambda: glb.convert(character, animations=rgis_clips(character))),
        ("rgis_clip.gltf", lambda: gltf.convert(character, animations=rgis_clips(character))),
    ]


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    """Validate every export in one Node run; the start-up dominates."""
    folder = tmp_path_factory.mktemp("khronos")
    paths = []
    for name, build in exports():
        path = folder / name
        path.write_bytes(build())
        paths.append(path)
    return {path.name: report for path, report in zip(paths, khronos.validate(*paths))}


@pytest.mark.parametrize("name", [name for name, _ in exports()])
def test_the_khronos_validator_accepts_the_export(name, reports):
    report = reports[name]
    assert report["numErrors"] == 0, khronos.summary(report)
    assert report["numWarnings"] == 0, khronos.summary(report)
    assert khronos.unexpected(report) == [], khronos.summary(report)


def test_the_validator_is_really_judging(tmp_path):
    """
    A broken file must come back with errors, or a green run means nothing.

    Pointing the skin's inverse bind matrices past the end of the accessors
    is a plain spec violation the validator is documented to report.
    """
    import json
    import struct

    payload = bytearray(glb.convert(asymmetric_character()))
    json_length = struct.unpack_from("<I", payload, 12)[0]
    document = json.loads(payload[20 : 20 + json_length])
    document["skins"][0]["inverseBindMatrices"] = len(document["accessors"]) + 5
    text = json.dumps(document).encode()
    text += b" " * (-len(text) % 4)
    rest = payload[20 + json_length :]
    broken = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(text) + len(rest))
    broken += struct.pack("<I", len(text)) + b"JSON" + text + rest

    path = tmp_path / "broken.glb"
    path.write_bytes(broken)
    (report,) = khronos.validate(path)
    assert report["numErrors"] > 0, khronos.summary(report)
