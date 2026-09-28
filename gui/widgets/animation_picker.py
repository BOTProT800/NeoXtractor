"""A dialog for picking one or more NeoX animation files out of an NPK."""

from PySide6 import QtCore, QtWidgets


class AnimationPicker(QtWidgets.QDialog):
    """
    Choose animation entries to attach to a mesh.

    Playable characters draw on a shared action library with thousands of
    single-clip files in one folder, so the list is filterable and allows a
    multiple selection; NPCs have a single file and the common case stays one
    click.
    """

    def __init__(self, names: list[str], parent=None, allow_browse: bool = True):
        super().__init__(parent)
        self.setWindowTitle("Choose animations")
        self.resize(560, 460)

        self._browse_requested = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(
            QtWidgets.QLabel(
                f"{len(names)} animation file(s) found next to this mesh. "
                "Select one or more."
            )
        )

        self._filter = QtWidgets.QLineEdit(self)
        self._filter.setPlaceholderText("Filter by name, e.g. walk or crouch")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self._filter)

        self._list = QtWidgets.QListWidget(self)
        self._list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        for name in names:
            self._list.addItem(QtWidgets.QListWidgetItem(name))
        if names:
            self._list.item(0).setSelected(True)
        self._list.itemDoubleClicked.connect(lambda _item: self.accept())
        layout.addWidget(self._list, 1)

        self._count = QtWidgets.QLabel(self)
        self._list.itemSelectionChanged.connect(self._update_count)
        layout.addWidget(self._count)

        buttons = QtWidgets.QHBoxLayout()
        select_visible = QtWidgets.QPushButton("Select all shown", self)
        select_visible.clicked.connect(self._select_visible)
        buttons.addWidget(select_visible)

        clear = QtWidgets.QPushButton("Clear selection", self)
        clear.clicked.connect(self._list.clearSelection)
        buttons.addWidget(clear)

        if allow_browse:
            browse = QtWidgets.QPushButton("Browse on disk...", self)
            browse.clicked.connect(self._browse)
            buttons.addWidget(browse)

        buttons.addStretch(1)
        layout.addLayout(buttons)

        box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)

        self._update_count()

    def _apply_filter(self, text: str) -> None:
        needle = text.strip().lower()
        for row in range(self._list.count()):
            item = self._list.item(row)
            item.setHidden(bool(needle) and needle not in item.text().lower())

    def _select_visible(self) -> None:
        for row in range(self._list.count()):
            item = self._list.item(row)
            if not item.isHidden():
                item.setSelected(True)

    def _update_count(self) -> None:
        self._count.setText(f"{len(self._list.selectedItems())} selected")

    def _browse(self) -> None:
        self._browse_requested = True
        self.accept()

    @property
    def browse_requested(self) -> bool:
        """True when the user asked for a file on disk instead."""
        return self._browse_requested

    def selected_names(self) -> list[str]:
        """Entry names the user picked, in list order."""
        if self._browse_requested:
            return []
        chosen = {item.text() for item in self._list.selectedItems()}
        return [
            self._list.item(row).text()
            for row in range(self._list.count())
            if self._list.item(row).text() in chosen
        ]


def pick_animations(parent, names: list[str]) -> tuple[list[str] | None, bool]:
    """
    Run the picker.

    Returns:
    - ``(names, browse)``. ``names`` is None when the user cancelled;
      ``browse`` is True when they asked to pick a file from disk instead.
    """
    dialog = AnimationPicker(names, parent)
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None, False
    if dialog.browse_requested:
        return None, True
    chosen = dialog.selected_names()
    if not chosen:
        return None, False
    return chosen, False
