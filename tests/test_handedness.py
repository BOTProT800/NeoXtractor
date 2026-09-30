"""
The rig's own left/right names against the way it faces.

This is the check that does not share the exporter's idea of the data, and
the one that showed NeoX to be left-handed. The fixture below is laid out the
way the real player rig was (bones named left at -X, toes towards +Z, in NeoX
coordinates), with round numbers rather than the game's.
"""

import numpy as np
import pytest

from core.mesh_converter.gltf_scene import build_scene
from core.mesh_converter.handedness import check_handedness, side_of
from core.mesh_converter.skeleton import IDENTITY_CONVERSION
from tests.support.synthetic import make_mesh_data, row_vector_matrix


@pytest.mark.parametrize(
    "name, expected",
    [
        ("biped_l_hand", ("l", "biped_hand")),
        ("biped_r_toe0", ("r", "biped_toe0")),
        ("Bip01 L Foot", ("l", "bip01_foot")),
        ("hand_right", ("r", "hand")),
        ("biped_spine", None),
        ("l", None),
        ("l_r_both", None),
    ],
)
def test_bone_names_are_read_for_their_side(name, expected):
    assert side_of(name) == expected


def biped_rig():
    """Feet, toes and hands on both sides, left at -X, facing +Z (NeoX)."""
    bones = [
        ("biped", None, (0.0, 9.5, 0.5)),
        ("biped_l_foot", "biped", (-1.7, 1.0, -0.1)),
        ("biped_l_toe0", "biped_l_foot", (-2.1, 0.1, 1.2)),
        ("biped_r_foot", "biped", (1.7, 1.0, -0.1)),
        ("biped_r_toe0", "biped_r_foot", (2.1, 0.1, 1.2)),
        ("biped_l_hand", "biped", (-4.5, 11.0, 1.3)),
        ("biped_r_hand", "biped", (4.5, 11.0, 1.3)),
    ]
    index_of = {name: index for index, (name, _, _) in enumerate(bones)}
    return make_mesh_data(
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
        bone_parents=[-1 if parent is None else index_of[parent] for _, parent, _ in bones],
        bone_names=[name for name, _, _ in bones],
        bone_matrices=[row_vector_matrix(origin) for _, _, origin in bones],
        joints=[(0, 0, 0, 0)] * 3,
        weights=[(1.0, 0.0, 0.0, 0.0)] * 3,
    )


def test_the_default_export_shows_the_rig_as_modelled():
    scene = build_scene(biped_rig())
    notes = [note for note in scene.diagnostics if note.startswith("handedness")]
    assert notes and "as modelled" in notes[0], scene.diagnostics


def test_an_unmirrored_export_is_called_a_mirror_image():
    """What the exporter did from 27 to 30 September 2026."""
    scene = build_scene(biped_rig(), conversion=IDENTITY_CONVERSION)
    notes = [note for note in scene.diagnostics if note.startswith("handedness")]
    assert notes and "mirror image" in notes[0], scene.diagnostics
    assert "shows the mirror image of the model" in notes[0]


def test_the_measure_itself():
    """Right-handed, Y up, facing +Z: left is +X."""
    names = ["foot_l", "toe_l", "foot_r", "toe_r"]
    as_modelled = [(1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (-1.0, 1.0, 0.0), (-1.0, 0.0, 1.0)]
    report = check_handedness(names, as_modelled)
    assert report.agreement == pytest.approx(1.0)
    assert report.verdict == "as modelled"
    assert report.pairs == 2

    mirrored = np.asarray(as_modelled) * np.array([-1.0, 1.0, 1.0])
    assert check_handedness(names, mirrored).verdict == "mirror image"


def test_a_rig_that_cannot_tell_says_nothing():
    # No side-named bones at all.
    assert check_handedness(["root", "spine"], [(0, 0, 0), (0, 1, 0)]) is None
    # Sides, but nothing that says which way the character faces.
    assert check_handedness(["hand_l", "hand_r"], [(1, 1, 0), (-1, 1, 0)]) is None
