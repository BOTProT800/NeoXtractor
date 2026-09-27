"""Code for viewer tab window customization."""

import os
import posixpath
from pathlib import Path
from typing import TYPE_CHECKING, cast

from PySide6 import QtCore, QtWidgets

from core.anim_loader import is_rgis, read_rgis
from core.logger import get_logger
from core.mesh_converter import FORMATS, convert_mesh
from core.mesh_converter.animation import clips_from_rgis
from core.mesh_converter.formats import glb
from core.mesh_converter.gltf_scene import build_scene
from gui.utils.npk import get_npk_file
from gui.widgets.managed_rhi_widget import ManagedRhiWidget

if TYPE_CHECKING:
    from gui.widgets.viewers.mesh_viewer.viewer_widget import MeshViewer
    from gui.windows.viewer_tab_window import ViewerTabWindow


BROWSE_LABEL = "Browse for a .gis file on disk..."


def _sibling_animations(viewer: "MeshViewer") -> list[tuple[str, int]]:
    """
    Find ``.gis`` entries sitting in the same NPK folder as the open mesh.

    A mesh and the animations that drive it ship together: in the samples,
    25 of 26 NPC meshes have a ``.gis`` in their own folder, and player
    characters keep a shared library folder in the same NPK. So the file is
    almost always already open, and there is no reason to make the user go
    hunting for it on disk.

    Returns:
    - ``(entry name, row)`` pairs, sorted by name.
    """
    entry = viewer.get_file()
    npk = get_npk_file()
    if entry is None or npk is None:
        return []

    mesh_name = (getattr(entry, "filename", "") or "").replace("\\", "/")
    folder = posixpath.dirname(mesh_name)

    found = []
    for row, index in enumerate(npk.indices):
        name = (getattr(index, "filename", "") or "").replace("\\", "/")
        if name.lower().endswith(".gis") and posixpath.dirname(name) == folder:
            found.append((name, row))
    return sorted(found)


def _read_npk_entry(row: int) -> bytes | None:
    """Read one NPK entry's bytes, loading it if the list has not yet."""
    npk = get_npk_file()
    if npk is None:
        return None
    entry = npk.entries.get(row)
    if entry is None or not entry.data:
        with open(npk.file_path, "rb") as handle:
            npk.load_entry(row, handle)
        entry = npk.entries.get(row)
    return entry.data if entry is not None else None


def _load_animation_clips(window, viewer: "MeshViewer"):
    """
    Pick a ``.gis`` and convert its clips onto the loaded mesh.

    Prefers animation files sitting beside the mesh in the open NPK, and falls
    back to a file dialog when there are none.

    Returns:
    - ``(clips, error)``. ``clips`` is None when the user cancelled.
    """
    mesh = viewer.render_widget.mesh_data
    if mesh is None:
        return None, "no mesh loaded"

    siblings = _sibling_animations(viewer)
    payload: bytes | None = None
    source_name = ""

    if siblings:
        labels = [name for name, _ in siblings] + [BROWSE_LABEL]
        choice, accepted = QtWidgets.QInputDialog.getItem(
            window,
            "Choose an animation file",
            f"{len(siblings)} animation file(s) found next to this mesh:",
            labels,
            0,
            False,
        )
        if not accepted:
            return None, None
        if choice != BROWSE_LABEL:
            row = dict(siblings)[choice]
            payload = _read_npk_entry(row)
            source_name = choice
            if payload is None:
                return None, f"could not read {choice} out of the open NPK"

    if payload is None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            window,
            "Select the NeoX animation file for this mesh",
            "",
            "NeoX animation (*.gis);;All files (*)",
        )
        if not file_path:
            return None, None
        payload = Path(file_path).read_bytes()
        source_name = Path(file_path).name

    try:
        if not is_rgis(payload):
            return None, (
                f"{source_name} is not a NeoX RGIS animation file "
                f"(it starts with {payload[:4]!r})"
            )
        scene = build_scene(mesh.raw_data)
        if scene.skeleton is None:
            return None, "this mesh has no usable skeleton, so clips cannot be attached"

        rgis = read_rgis(payload)
        clips, notes = clips_from_rgis(rgis, scene.skeleton)
    except Exception as error:  # noqa: BLE001 - surfaced to the user verbatim
        get_logger().exception("Failed to read animations from %s", source_name)
        return None, str(error) or error.__class__.__name__

    for name, reason in rgis.skipped:
        get_logger().warning("Animation %s skipped: %s", name, reason)
    for note in notes:
        get_logger().warning("Animation: %s", note)

    if not clips:
        detail = "; ".join(reason for _, reason in rgis.skipped) or "; ".join(notes)
        return None, f"no clip in that file can drive this rig. {detail}"
    return clips, None


def _save_as_format(viewer: "MeshViewer", target_format, file_path: str, **options):
    """
    Save the current mesh in the specified format.

    Parameters:
    - viewer: The MeshViewer instance containing the mesh to save.
    - target_format: The format to save the mesh as.
    - file_path: Destination path.

    Returns:
    - None on success, or the error message when the conversion failed.
    """
    mesh = viewer.render_widget.mesh_data
    if mesh is None:
        return "no mesh loaded"

    # Convert before touching the filesystem: opening the file first leaves an
    # empty file behind whenever the conversion rejects the data.
    try:
        payload = convert_mesh(mesh.raw_data, target_format, **options)
    except Exception as error:  # noqa: BLE001 - surfaced to the user verbatim
        get_logger().exception("Failed to convert mesh to %s", target_format.NAME)
        return str(error) or error.__class__.__name__

    try:
        with open(file_path, "wb") as f:
            f.write(payload)
    except OSError as error:
        get_logger().exception("Failed to write %s", file_path)
        return str(error)
    return None


def _save_current_as_format(window: "ViewerTabWindow", target_format):
    """
    Save the current mesh in the specified format.

    Parameters:
    - target_format: The format to save the mesh as.
    """
    viewer = cast("MeshViewer", window.tab_widget.currentWidget())
    if viewer is None:
        QtWidgets.QMessageBox.warning(
            window, "No File Opened", "Please open a mesh file before saving."
        )
        return
    file_dialog = QtWidgets.QFileDialog()
    file_path, _ = file_dialog.getSaveFileName(
        None,
        f"Save Mesh as {target_format.NAME}",
        "",
        f"{target_format.NAME} Files (*{target_format.EXTENSION})",
    )
    if file_path:
        error = _save_as_format(viewer, target_format, file_path)
        if error is None:
            QtWidgets.QMessageBox.information(
                window,
                "Save Successful",
                f"Mesh saved successfully as {target_format.NAME}.",
            )
        else:
            QtWidgets.QMessageBox.critical(
                window,
                "Save Failed",
                f"Could not save the mesh as {target_format.NAME}.\n\n{error}",
            )


def _save_current_with_animations(window: "ViewerTabWindow"):
    """Save the current mesh as GLB with clips from a chosen ``.gis`` file."""
    viewer = cast("MeshViewer", window.tab_widget.currentWidget())
    if viewer is None or viewer.render_widget.mesh_data is None:
        QtWidgets.QMessageBox.warning(
            window, "No File Opened", "Please open a mesh file before saving."
        )
        return

    clips, error = _load_animation_clips(window, viewer)
    if error is not None:
        QtWidgets.QMessageBox.critical(window, "Animations Not Loaded", error)
        return
    if clips is None:
        return

    file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
        window,
        f"Save Mesh with {len(clips)} animation(s)",
        "",
        f"{glb.NAME} Files (*{glb.EXTENSION})",
    )
    if not file_path:
        return

    failure = _save_as_format(viewer, glb, file_path, animations=clips)
    if failure is None:
        QtWidgets.QMessageBox.information(
            window,
            "Save Successful",
            f"Saved with {len(clips)} animation(s): "
            + ", ".join(clip.name for clip in clips[:8])
            + ("..." if len(clips) > 8 else ""),
        )
    else:
        QtWidgets.QMessageBox.critical(
            window,
            "Save Failed",
            f"Could not save the mesh.\n\n{failure}",
        )


def _save_all_as_format(window: "ViewerTabWindow", target_format):
    """
    Save all currently opened meshes in the specified format.

    Parameters:
    - target_format: The format to save the meshes as.
    """
    file_count = window.tab_widget.count()
    if file_count == 0:
        QtWidgets.QMessageBox.warning(
            window, "No Files Opened", "Please open mesh files before saving all."
        )
        return
    save_directory = QtWidgets.QFileDialog.getExistingDirectory(
        None, f"Save All Meshes as {target_format.NAME}", ""
    )
    if not save_directory:
        return

    saved = 0
    failures: list[str] = []
    for i in range(window.tab_widget.count()):
        viewer = cast("MeshViewer", window.tab_widget.widget(i))
        if viewer is None or viewer.render_widget.mesh_data is None:
            continue
        tab_name = os.path.splitext(window.tab_widget.tabText(i))[0]
        file_path = os.path.join(
            save_directory,
            f"{tab_name}{target_format.EXTENSION}",
        )
        error = _save_as_format(viewer, target_format, file_path)
        if error is None:
            saved += 1
        else:
            failures.append(f"{tab_name}: {error}")

    summary = f"{saved} mesh(es) saved as {target_format.NAME}."
    if not failures:
        QtWidgets.QMessageBox.information(window, "Save All Successful", summary)
        return

    shown = "\n".join(failures[:10])
    if len(failures) > 10:
        shown += f"\n... and {len(failures) - 10} more"
    QtWidgets.QMessageBox.warning(
        window,
        "Save All Finished With Errors",
        f"{summary}\n\n{len(failures)} failed:\n{shown}",
    )


class _EventFilter(QtCore.QObject):
    """Event filter for the mesh viewer tab window. Removes the surface type setter after shown."""

    def __init__(self, window: "ViewerTabWindow", setter: ManagedRhiWidget):
        super().__init__(window)
        self._window = window
        self._setter = setter

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.Show:
            self._window.central_layout.removeWidget(self._setter)
            return True
        return False


def setup_mesh_viewer_tab_window(window: "ViewerTabWindow"):
    """Setup the mesh viewer tab window."""

    # Forces the tab window to use current graphics backend.
    surface_type_setter = ManagedRhiWidget()
    window.central_layout.addWidget(surface_type_setter)

    event_filter = _EventFilter(window, surface_type_setter)
    window.installEventFilter(event_filter)

    save_as_menu = window.menuBar().addMenu("Save As")

    for fmt in FORMATS:
        action = save_as_menu.addAction(fmt.NAME)
        action.triggered.connect(
            lambda _, fmt=fmt: _save_current_as_format(window, fmt)
        )

    save_as_menu.addSeparator()
    animated_action = save_as_menu.addAction(
        f"{glb.NAME} with animations (.gis)..."
    )
    animated_action.triggered.connect(
        lambda _: _save_current_with_animations(window)
    )

    save_all_as_menu = window.menuBar().addMenu("Save All As")
    for fmt in FORMATS:
        action = save_all_as_menu.addAction(fmt.NAME)
        action.triggered.connect(lambda _, fmt=fmt: _save_all_as_format(window, fmt))
