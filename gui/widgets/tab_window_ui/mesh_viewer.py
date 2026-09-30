"""Code for viewer tab window customization."""

import os
import posixpath
from pathlib import Path
from typing import TYPE_CHECKING, cast

from PySide6 import QtCore, QtWidgets

from core.anim_loader import RGISReadError, read_gis
from core.logger import get_logger
from core.mesh_converter import FORMATS, convert_mesh
from core.mesh_converter.animation import clips_from_rgis
from core.mesh_converter.formats import glb
from core.mesh_converter.gltf_scene import build_scene
from gui.utils.npk import get_npk_file
from gui.widgets.animation_picker import pick_animations
from gui.widgets.managed_rhi_widget import ManagedRhiWidget

if TYPE_CHECKING:
    from gui.widgets.viewers.mesh_viewer.viewer_widget import MeshViewer
    from gui.windows.viewer_tab_window import ViewerTabWindow


def _npk_animations(viewer: "MeshViewer") -> tuple[list[tuple[str, int]], int]:
    """
    Every ``.gis`` entry in the open NPK, the ones beside the mesh first.

    A mesh and the animations that drive it ship in the same NPK, but not
    always in the same folder. In the samples 25 of 26 NPC meshes have a
    ``.gis`` in their own folder, while playable characters draw on a shared
    library elsewhere in the NPK (``common/dongzuoku_gis/``,
    ``<character>/common_gis/``; 2402 files in ``male.npk``). Offering only
    the mesh's folder left a player model with nothing to pick, so it could
    only be exported without animations.

    Returns:
    - ``(entries, beside)``: ``(entry name, row)`` pairs, of which the first
      ``beside`` come from the mesh's own folder. Each group is sorted by name.
    """
    entry = viewer.get_file()
    npk = get_npk_file()
    if entry is None or npk is None:
        return [], 0

    mesh_name = (getattr(entry, "filename", "") or "").replace("\\", "/")
    folder = posixpath.dirname(mesh_name)

    beside, elsewhere = [], []
    for row, index in enumerate(npk.indices):
        name = (getattr(index, "filename", "") or "").replace("\\", "/")
        if name.lower().endswith(".gis"):
            group = beside if posixpath.dirname(name) == folder else elsewhere
            group.append((name, row))
    return sorted(beside) + sorted(elsewhere), len(beside)


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


def _clips_from_sources(sources, skeleton):
    """
    Read every chosen animation source and merge their clips.

    Parameters:
    - sources: ``(label, bytes)`` pairs.
    - skeleton: the rig the clips must drive.

    Returns:
    - ``(clips, notes)``. Clip names are made unique across sources, because
      the shared library stores one clip per file and two files can name their
      clip the same thing.
    """
    clips = []
    notes: list[str] = []
    taken: set[str] = set()

    for label, payload in sources:
        stem = posixpath.splitext(posixpath.basename(label.replace("\\", "/")))[0]
        try:
            parsed = read_gis(payload)
        except RGISReadError as error:
            notes.append(f"{stem}: {error}")
            continue

        for name, reason in parsed.skipped:
            notes.append(f"{stem}/{name}: {reason}")

        found, source_notes = clips_from_rgis(parsed, skeleton)
        notes.extend(f"{stem}: {note}" for note in source_notes)

        for clip in found:
            name = clip.name or stem
            if name in taken:
                name = f"{stem}_{clip.name}"
                suffix = 2
                while name in taken:
                    name = f"{stem}_{clip.name}_{suffix}"
                    suffix += 1
                notes.append(f"renamed a duplicate clip to {name!r}")
            clip.name = name
            taken.add(name)
            clips.append(clip)

    return clips, notes


def _load_animation_clips(window, viewer: "MeshViewer"):
    """
    Pick one or more ``.gis`` files and convert their clips onto the mesh.

    Offers every animation file in the open NPK, those beside the mesh first
    and selected, and falls back to a file dialog when there are none or the
    user asks for one.

    Returns:
    - ``(clips, error)``. ``clips`` is None when the user cancelled.
    """
    mesh = viewer.render_widget.mesh_data
    if mesh is None:
        return None, "no mesh loaded"

    try:
        scene = build_scene(mesh.raw_data)
    except Exception as error:  # noqa: BLE001 - surfaced to the user verbatim
        get_logger().exception("Failed to build the skeleton for animation export")
        return None, str(error) or error.__class__.__name__
    if scene.skeleton is None:
        return None, "this mesh has no usable skeleton, so clips cannot be attached"

    sources: list[tuple[str, bytes]] = []
    available, beside = _npk_animations(viewer)
    browse = not available

    if available:
        chosen, browse = pick_animations(
            window, [name for name, _ in available], beside=beside
        )
        if chosen is None and not browse:
            return None, None
        if chosen:
            rows = dict(available)
            for name in chosen:
                payload = _read_npk_entry(rows[name])
                if payload is None:
                    return None, f"could not read {name} out of the open NPK"
                sources.append((name, payload))

    if browse:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            window,
            "Select NeoX animation files for this mesh",
            "",
            "NeoX animation (*.gis);;All files (*)",
        )
        if not paths:
            return None, None
        for path in paths:
            sources.append((Path(path).name, Path(path).read_bytes()))

    if not sources:
        return None, None

    try:
        clips, notes = _clips_from_sources(sources, scene.skeleton)
    except Exception as error:  # noqa: BLE001 - surfaced to the user verbatim
        get_logger().exception("Failed to read animations")
        return None, str(error) or error.__class__.__name__

    for note in notes:
        get_logger().warning("Animation: %s", note)

    if not clips:
        detail = "; ".join(notes[:4]) or "the files hold no clip this rig can use"
        return None, f"nothing could be attached. {detail}"
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
