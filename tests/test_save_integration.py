"""
Tests for the save path used by the viewer's "Save As" menu.

``_save_as_format`` is exercised with a stub viewer so no Qt event loop is
needed; the message boxes live in the callers, which is why this function
returns an error string instead of showing one itself.
"""

import numpy as np
import pytest

from core.mesh_converter import FORMATS
from core.mesh_converter.formats import glb
from tests.support.synthetic import asymmetric_character, make_mesh_data, row_vector_matrix


class _StubRenderWidget:
    def __init__(self, mesh_data):
        self.mesh_data = mesh_data


class _StubMeshData:
    def __init__(self, raw_data):
        self.raw_data = raw_data


class _StubViewer:
    def __init__(self, mesh=None):
        self.render_widget = _StubRenderWidget(
            None if mesh is None else _StubMeshData(mesh)
        )


@pytest.fixture
def save_as_format():
    """Import the save helper, skipping if PySide6 is unavailable."""
    pytest.importorskip("PySide6")
    from gui.widgets.tab_window_ui.mesh_viewer import _save_as_format

    return _save_as_format


def test_a_successful_save_writes_the_payload(save_as_format, tmp_path):
    mesh = asymmetric_character()
    destination = tmp_path / "character.glb"

    error = save_as_format(_StubViewer(mesh), glb, str(destination))

    assert error is None
    assert destination.read_bytes() == glb.convert(mesh)


def test_a_failed_conversion_leaves_no_file_behind(save_as_format, tmp_path):
    """
    Opening the file first left an empty one whenever conversion failed.

    The conversion now runs before the filesystem is touched, so a rejected
    mesh produces a reported error and no output at all.
    """
    broken = make_mesh_data(
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
        bone_parents=[-1],
        bone_names=["root"],
        bone_matrices=[row_vector_matrix((0.0, 0.0, 0.0))],
        joints=[(0, 0, 0, 0)] * 3,
        weights=[(0.25, 0.0, 0.0, 0.0)] * 3,
    )
    destination = tmp_path / "broken.glb"

    error = save_as_format(_StubViewer(broken), glb, str(destination))

    assert error is not None
    assert "wrong offset" in error
    assert not destination.exists()


def test_no_mesh_loaded_is_reported(save_as_format, tmp_path):
    destination = tmp_path / "nothing.glb"

    error = save_as_format(_StubViewer(None), glb, str(destination))

    assert error == "no mesh loaded"
    assert not destination.exists()


def test_every_registered_format_has_a_name_and_extension():
    for module in FORMATS:
        assert isinstance(module.NAME, str) and module.NAME
        assert module.EXTENSION.startswith(".")
        assert callable(module.convert)


class _StubEntry:
    """Minimal stand-in for an NPK entry: the viewer only reads its name."""

    def __init__(self, filename):
        self.filename = filename


class _StubIndex:
    def __init__(self, filename):
        self.filename = filename


class _StubNPK:
    def __init__(self, names):
        self.indices = [_StubIndex(name) for name in names]


class _ViewerWithFile(_StubViewer):
    def __init__(self, mesh, entry_name):
        super().__init__(mesh)
        self._entry = _StubEntry(entry_name)

    def get_file(self):
        return self._entry


@pytest.fixture
def sibling_animations(monkeypatch):
    """Import the discovery helper with a stubbed NPK accessor."""
    pytest.importorskip("PySide6")
    from gui.widgets.tab_window_ui import mesh_viewer as module

    def install(names):
        monkeypatch.setattr(module, "get_npk_file", lambda: _StubNPK(names))
        return module._sibling_animations

    return install


#: NPK entry paths use backslashes; built here so no escape can hide in a
#: literal.
SEP = chr(92)


def npk_path(folder: str, name: str) -> str:
    """An NPK-style entry path, ``folder\\name``."""
    return folder + SEP + name


def test_animations_beside_the_mesh_are_found(sibling_animations):
    """
    A mesh and its animations ship in the same NPK folder.

    Measured on the samples: 25 of 26 NPC meshes have a .gis in their own
    folder, so the file is almost always already open and there is no reason
    to send the user hunting on disk.
    """
    mesh_entry = npk_path("tiejiayong", "tiejiayong_03.mesh")
    find = sibling_animations(
        [
            mesh_entry,
            npk_path("tiejiayong", "tiejiayong.gis"),
            npk_path("tiejiayong", "tiejiayong_d.dds"),
            npk_path("otherguy", "otherguy.gis"),
        ]
    )

    found = find(_ViewerWithFile(asymmetric_character(), mesh_entry))

    # Backslash paths are normalised on the way out.
    assert found == [("tiejiayong/tiejiayong.gis", 1)]


def test_animations_in_other_folders_are_not_offered(sibling_animations):
    mesh_entry = npk_path("a", "thing.mesh")
    find = sibling_animations(
        [mesh_entry, npk_path("b", "one.gis"), npk_path("c", "two.gis")]
    )

    assert find(_ViewerWithFile(asymmetric_character(), mesh_entry)) == []


def test_several_siblings_come_back_sorted(sibling_animations):
    """Player characters keep a shared library, so several is the normal case."""
    mesh_entry = npk_path("lib", "hero.mesh")
    find = sibling_animations(
        [
            mesh_entry,
            npk_path("lib", "run.gis"),
            npk_path("lib", "attack.gis"),
            npk_path("lib", "idle.gis"),
        ]
    )

    found = find(_ViewerWithFile(asymmetric_character(), mesh_entry))

    assert [name for name, _ in found] == [
        "lib/attack.gis",
        "lib/idle.gis",
        "lib/run.gis",
    ]


class TestMergingSeveralAnimationFiles:
    """
    Playable characters draw on a library of single-clip files, so combining
    several into one GLB is the normal case there.
    """

    def merge(self, sources, skeleton):
        pytest.importorskip("PySide6")
        from gui.widgets.tab_window_ui.mesh_viewer import _clips_from_sources

        return _clips_from_sources(sources, skeleton)

    def rig(self):
        from core.mesh_converter.gltf_scene import build_scene

        mesh = asymmetric_character()
        return mesh, build_scene(mesh).skeleton

    def one_clip_file(self, clip_name, degrees):
        from tests.support.synthetic import build_rgis_file

        identity = (0.0, 0.0, 0.0, 1.0)
        turn = (0.0, 0.0, np.sin(np.radians(degrees / 2)), np.cos(np.radians(degrees / 2)))
        return build_rgis_file(
            reference=[("arm_l", (1.0, 2.0, 0.0), identity, (1.0, 1.0, 1.0))],
            clips=[
                {
                    "name": clip_name,
                    "times": [0.0, 1.0],
                    "tracks": [
                        ("arm_l", (1.0, 2.0, 0.0), [identity, turn], (1.0, 1.0, 1.0))
                    ],
                }
            ],
        )

    def test_clips_from_several_files_are_combined(self):
        mesh, skeleton = self.rig()
        sources = [
            ("lib/walk_f.gis", self.one_clip_file("walk_f", 30)),
            ("lib/attack.gis", self.one_clip_file("attack", 60)),
            ("lib/idle.gis", self.one_clip_file("idle", 10)),
        ]

        clips, notes = self.merge(sources, skeleton)

        assert [clip.name for clip in clips] == ["walk_f", "attack", "idle"]
        assert notes == []

    def test_duplicate_clip_names_are_disambiguated(self):
        """Two library files can each hold a clip called the same thing."""
        mesh, skeleton = self.rig()
        sources = [
            ("lib/smg_idle.gis", self.one_clip_file("idle", 30)),
            ("lib/rifle_idle.gis", self.one_clip_file("idle", 60)),
        ]

        clips, notes = self.merge(sources, skeleton)

        names = [clip.name for clip in clips]
        assert names[0] == "idle"
        assert names[1] == "rifle_idle_idle"
        assert len(set(names)) == 2
        assert any("renamed" in note for note in notes)

    def test_a_bad_file_is_reported_and_the_rest_still_load(self):
        mesh, skeleton = self.rig()
        sources = [
            ("lib/good.gis", self.one_clip_file("good", 30)),
            ("lib/broken.gis", b"not an animation at all"),
        ]

        clips, notes = self.merge(sources, skeleton)

        assert [clip.name for clip in clips] == ["good"]
        assert any("broken" in note for note in notes)

    def test_the_merged_clips_export_into_one_glb(self):
        from core.mesh_converter.formats import glb
        from tests.support.gltf_reader import read_any

        mesh, skeleton = self.rig()
        sources = [
            ("lib/walk_f.gis", self.one_clip_file("walk_f", 30)),
            ("lib/attack.gis", self.one_clip_file("attack", 60)),
        ]
        clips, _ = self.merge(sources, skeleton)

        document = read_any(glb.convert(mesh, animations=clips))

        assert [a["name"] for a in document.json_data["animations"]] == [
            "walk_f",
            "attack",
        ]
