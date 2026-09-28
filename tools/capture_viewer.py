"""
Photograph NeoXtractor's own mesh viewer, the way a user would see a model.

The viewer and the exporters must show the same model, and the model the game
shows. They once disagreed: the viewer mirrored X long after the GLB export
stopped doing it, and nothing but a picture could have said so. This takes the
picture, so the viewer can be put next to ``tools/render_glb.py`` and the game.

Run from the repository root::

    uv run python tools/capture_viewer.py modelo.mesh
    uv run python tools/capture_viewer.py modelo.mesh --out vista.png --size 480

On Linux without a screen, wrap it in ``xvfb-run -a``. It writes
``modelo_viewer.png`` next to the input: the views the viewer's keys 1, 3,
Ctrl+1 and Ctrl+7 give, with the skeleton drawn. Each caption says where the
camera sits, read from the camera's own matrix. Key 1 looks from +Z, as the
front view of ``tools/render_glb.py`` does, so the two pictures should match
side by side.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The viewer looks for its compiled shaders next to the main script.
sys.argv[0] = str(REPO_ROOT / "main.py")

from PySide6 import QtGui, QtWidgets  # noqa: E402

#: (caption, direction, opposite) as the viewer's own keys set them.
VIEWS = (
    ("key 1", "FRONT", False),
    ("key 3", "RIGHT", False),
    ("key Ctrl+1", "FRONT", True),
    ("key Ctrl+7", "TOP", True),
)


def camera_side(camera) -> str:
    """Where the camera sits, read off its own matrix rather than its label."""
    toward = camera.rot().inverted()[0].map(QtGui.QVector4D(0.0, 0.0, 1.0, 0.0))
    components = (toward.x(), toward.y(), toward.z())
    axis = max(range(3), key=lambda index: abs(components[index]))
    sign = "+" if components[axis] > 0 else "-"
    return f"camera on {sign}{'XYZ'[axis]}"


def capture(mesh, size: int = 480) -> list[tuple[str, QtGui.QImage]]:
    """
    Draw ``mesh`` in a real viewer widget; return one picture per view.

    A ``QApplication`` must already exist.
    """
    from gui.widgets.viewers.mesh_viewer.camera import OrthogonalDirection
    from gui.widgets.viewers.mesh_viewer.render_widget import MeshRenderWidget

    widget = MeshRenderWidget()
    widget.draw_text = False
    widget.draw_bones = True
    widget.resize(size, size)
    widget.show()
    widget.load_mesh(mesh)

    pictures = []
    for key, direction, opposite in VIEWS:
        widget.camera.orthogonal(OrthogonalDirection[direction], opposite)
        caption = f"{key}: {camera_side(widget.camera)}"
        # A couple of frames so the resources exist before the grab.
        for _ in range(3):
            QtWidgets.QApplication.processEvents()
        pictures.append((caption, widget.grabFramebuffer()))
    widget.close()
    return pictures


def sheet(pictures, title: str) -> QtGui.QImage:
    """Side by side, each with its caption."""
    width = sum(image.width() for _, image in pictures)
    height = max(image.height() for _, image in pictures)
    result = QtGui.QImage(width, height, QtGui.QImage.Format.Format_RGB32)
    result.fill(QtGui.QColor(40, 40, 40))
    painter = QtGui.QPainter(result)
    painter.setPen(QtGui.QColor(235, 235, 235))
    x = 0
    for caption, image in pictures:
        painter.drawImage(x, 0, image)
        painter.drawText(x + 8, 18, f"{title}  {caption}")
        x += image.width()
    painter.end()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mesh", help="path to the .mesh file")
    parser.add_argument("--out", help="where to write the picture")
    parser.add_argument("--size", type=int, default=480, help="pixels per view")
    args = parser.parse_args()

    application = QtWidgets.QApplication(sys.argv[:1])
    from core.mesh_loader import MeshLoader

    source = Path(args.mesh)
    mesh = MeshLoader().load_from_bytes(source.read_bytes())
    if mesh is None:
        print(f"the parser could not read {source}")
        return 1

    destination = Path(args.out) if args.out else source.with_name(source.stem + "_viewer.png")
    picture = sheet(capture(mesh, args.size), source.name)
    if not picture.save(str(destination)):
        print(f"could not write {destination}")
        return 1
    print(f"wrote {destination}")
    application.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
