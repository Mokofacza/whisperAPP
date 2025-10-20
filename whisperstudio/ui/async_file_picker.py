# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Iterable, Optional
from pathlib import Path
import sys

from PySide6 import QtCore, QtGui, QtWidgets

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".mka"}


def _default_start_dir() -> Optional[Path]:
    """Preferuj ./flat_audio (jeśli istnieje), potem Music, potem Home."""
    try:
        fa = (Path.cwd() / "flat_audio").resolve()
        if fa.exists() and fa.is_dir():
            return fa
    except Exception:
        pass
    # fallbacki systemowe
    start = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.MusicLocation)
    if start:
        p = Path(start)
        if p.exists():
            return p
    return Path.home()


class _AudioFilterProxy(QtCore.QSortFilterProxyModel):
    def __init__(self, suffixes: Iterable[str], parent=None):
        super().__init__(parent)
        self._suffixes = set(s.lower().lstrip(".") for s in suffixes)

    def filterAcceptsRow(self, source_row, source_parent):
        idx = self.sourceModel().index(source_row, 0, source_parent)
        if not idx.isValid():
            return False
        fs = self.sourceModel().fileInfo(idx)
        if fs.isDir():
            return True
        return fs.suffix().lower() in self._suffixes if self._suffixes else True


class FileSystemPicker(QtWidgets.QDialog):
    """
    Szybki, nienatywny selektor pliku/folderu z normalną NAWIGACJĄ:
    - dwuklik w folder => wchodzi do folderu
    - ⬆ (Up) => folder nadrzędny
    - lista dysków (Windows)
    - filtr rozszerzeń audio
    """

    def __init__(self, parent=None, mode: str = "file", suffixes: Iterable[str] = AUDIO_SUFFIXES):
        super().__init__(parent)
        assert mode in ("file", "dir")
        self.mode = mode
        self._setting_root = False  # by nie zapętlać ust. napędu
        style = self.style()  # Pobranie stylu dla ikon

        self.setWindowTitle("Wybierz plik" if mode == "file" else "Wybierz folder")
        self.resize(900, 560)
        self.setModal(True)

        v = QtWidgets.QVBoxLayout(self)

        # Pasek narzędzi: Dysk / Up / Ścieżka
        tb = QtWidgets.QHBoxLayout()
        self.drive = QtWidgets.QComboBox()
        self.btnUp = QtWidgets.QToolButton()
        self.btnUp.setIcon(style.standardIcon(QtWidgets.QStyle.SP_ArrowUp))
        self.pathEdit = QtWidgets.QLineEdit();
        self.pathEdit.setReadOnly(True)
        tb.addWidget(QtWidgets.QLabel("Dysk:"));
        tb.addWidget(self.drive)
        tb.addWidget(self.btnUp);
        tb.addWidget(self.pathEdit, 1)
        v.addLayout(tb)

        # Model FS
        self.model = QtWidgets.QFileSystemModel(self)
        self.model.setRootPath(QtCore.QDir.rootPath())
        if mode == "dir":
            self.model.setFilter(QtCore.QDir.AllDirs | QtCore.QDir.NoDotAndDotDot | QtCore.QDir.Drives)
        else:
            self.model.setFilter(QtCore.QDir.AllEntries | QtCore.QDir.NoDotAndDotDot | QtCore.QDir.Dirs)

        # Widok
        self.view = QtWidgets.QTreeView(self)
        self.view.setModel(self.model)
        self.view.setSortingEnabled(True)
        self.view.setExpandsOnDoubleClick(True)
        self.view.setAlternatingRowColors(True)
        self.view.setHeaderHidden(False)
        for i, w in enumerate((460, 120, 120, 160)):
            self.view.setColumnWidth(i, w)
        v.addWidget(self.view, 1)

        # Filtr plików audio w trybie "file"
        self.proxy = None
        if mode == "file":
            self.proxy = _AudioFilterProxy(suffixes, self)
            self.proxy.setSourceModel(self.model)
            self.view.setModel(self.proxy)
            self.view.sortByColumn(0, QtCore.Qt.AscendingOrder)  # Sortuj po nazwie

        # Pasek dołu: przyciski
        h = QtWidgets.QHBoxLayout()
        self.btnChoose = QtWidgets.QPushButton("Wybierz");
        self.btnChoose.setDefault(True)
        self.btnChoose.setIcon(style.standardIcon(QtWidgets.QStyle.SP_DialogOkButton))

        self.btnCancel = QtWidgets.QPushButton("Anuluj")
        self.btnCancel.setIcon(style.standardIcon(QtWidgets.QStyle.SP_DialogCancelButton))

        h.addStretch(1);
        h.addWidget(self.btnChoose);
        h.addWidget(self.btnCancel)
        v.addLayout(h)

        # Dyski (Windows)
        self._populate_drives()

        # Startowa lokalizacja — ./flat_audio, jeśli istnieje
        self._set_root(_default_start_dir())

        # Sygnały
        self.view.doubleClicked.connect(self._on_double_click)
        self.view.selectionModel().selectionChanged.connect(self._on_selection)
        self.btnChoose.clicked.connect(self.accept)
        self.btnCancel.clicked.connect(self.reject)
        self.btnUp.clicked.connect(self._go_up)
        self.drive.currentTextChanged.connect(self._go_drive)

        # Skróty
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Return), self, activated=self.accept)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Enter), self, activated=self.accept)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self, activated=self.reject)
        QtGui.QShortcut(QtGui.QKeySequence("Alt+Up"), self, activated=self._go_up)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Backspace), self, activated=self._go_up)

    # ---- API ----
    def selected_path(self) -> Optional[Path]:
        idx = self.view.currentIndex()
        if not idx.isValid():
            # Jeśli nic nie zaznaczono, a jesteśmy w trybie 'dir', zwróć obecny folder
            if self.mode == "dir":
                return Path(self.pathEdit.text())
            return None

        src = self.proxy.mapToSource(idx) if self.proxy else idx
        info = self.model.fileInfo(src)

        if self.mode == "dir":
            return Path(info.absoluteFilePath() if info.isDir() else info.absolutePath())

        # Tryb 'file'
        if info.isDir():
            return None  # Nie wybrano pliku
        return Path(info.absoluteFilePath())

    @staticmethod
    def pick_file(parent=None, suffixes=AUDIO_SUFFIXES) -> Optional[Path]:
        dlg = FileSystemPicker(parent, mode="file", suffixes=suffixes)
        return dlg.selected_path() if dlg.exec() == QtWidgets.QDialog.Accepted else None

    @staticmethod
    def pick_dir(parent=None) -> Optional[Path]:
        dlg = FileSystemPicker(parent, mode="dir")
        return dlg.selected_path() if dlg.exec() == QtWidgets.QDialog.Accepted else None

    # ---- Nawigacja / root ----
    def _set_root(self, path: Optional[Path]):
        if path is None:
            return
        try:
            path = path.resolve()
        except Exception:  # Np. niedostępny dysk
            return

        idx_src = self.model.index(str(path))
        idx = self.proxy.mapFromSource(idx_src) if self.proxy else idx_src

        if idx.isValid():
            self._setting_root = True
            try:
                self.view.setRootIndex(idx)
                self.pathEdit.setText(str(path))
                # Ustaw dysk na Windows
                if sys.platform.startswith("win"):
                    anchor = path.drive or path.anchor  # 'C:' lub 'C:\\'
                    if anchor and not anchor.endswith("\\"):
                        anchor = anchor + "\\"
                    self.drive.blockSignals(True)
                    # jeśli anchor jest na liście – ustaw; w innym razie nie zmieniaj
                    ix = self.drive.findText(anchor, QtCore.Qt.MatchStartsWith)
                    if ix >= 0:
                        self.drive.setCurrentIndex(ix)
                    self.drive.blockSignals(False)
            finally:
                self._setting_root = False
        else:
            # Jeśli ścieżka jest nieprawidłowa (np. nie istnieje), spróbuj rodzica
            if path.parent != path:
                self._set_root(path.parent)

    def _go_up(self):
        cur = Path(self.pathEdit.text()) if self.pathEdit.text() else None
        if not cur:
            return
        parent = cur.parent if cur.parent != cur else cur
        self._set_root(parent)

    def _go_drive(self, drv: str):
        if self._setting_root:
            return
        if not drv:
            return
        self._set_root(Path(drv))

    def _populate_drives(self):
        self.drive.clear()
        if sys.platform.startswith("win"):
            for d in QtCore.QDir.drives():
                self.drive.addItem(d.absolutePath(), d.absolutePath())
        else:
            # na *nix pokaż korzeń i home
            self.drive.addItem("/", "/")
            self.drive.addItem(str(Path.home()), str(Path.home()))

    # ---- Zdarzenia ----
    def _on_double_click(self, idx: QtCore.QModelIndex):
        src = self.proxy.mapToSource(idx) if self.proxy else idx
        info = self.model.fileInfo(src)
        if info.isDir():
            self._set_root(Path(info.absoluteFilePath()))
        else:
            # plik: akceptuj (tylko w trybie file)
            if self.mode == "file":
                self.accept()

    def _on_selection(self, *_):
        p = self.selected_path()
        if p and self.mode == "file":  # W trybie 'dir' nie aktualizuj ścieżki na zaznaczony folder
            self.pathEdit.setText(str(p))
        elif self.mode == "dir":
            # W trybie dir, ścieżka na górze zawsze pokazuje obecny folder (root)
            root_idx = self.view.rootIndex()
            src_root = self.proxy.mapToSource(root_idx) if self.proxy else root_idx
            info = self.model.fileInfo(src_root)
            self.pathEdit.setText(info.absoluteFilePath())