"""
Tests for the save path used by the viewer's "Save As" menu.

``_save_as_format`` is exercised with a stub viewer so no Qt event loop is
needed; the message boxes live in the callers, which is why this function
returns an error string instead of showing one itself.
"""

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
