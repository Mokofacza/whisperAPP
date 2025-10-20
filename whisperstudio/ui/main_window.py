from __future__ import annotations
from PySide6 import QtWidgets, QtGui
from .live_tab import LiveTab
from .files_tab import FilesTab
from .train_tab import TrainTab
from .merge_tab import MergeTab
from .model_selection import set_variant, get_variant, get_label

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper Studio")
        self.resize(1120, 760)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(LiveTab(), "LIVE")
        tabs.addTab(FilesTab(), "Pliki")
        tabs.addTab(TrainTab(), "Trening")
        tabs.addTab(MergeTab(), "Merge/Export")
        self.setCentralWidget(tabs)

        self._build_menu()
        self.statusBar().showMessage(f"Model: {get_label()}")

    def _build_menu(self):
        mb = self.menuBar()
        m = mb.addMenu("&Model")

        ag = QtGui.QActionGroup(self)
        ag.setExclusive(True)

        opts = [
            ("local",  "Whisper Small — Local"),
            ("lora",   "Whisper Small — LoRA"),
            ("merged", "Whisper Small — Merged"),
            ("full",   "Whisper Small — Full Trained"),
        ]

        current = get_variant()
        for key, title in opts:
            act = QtGui.QAction(title, self, checkable=True)
            act.setChecked(key == current)
            act.triggered.connect(lambda _=False, v=key: self._set_model(v))
            ag.addAction(act)
            m.addAction(act)

    def _set_model(self, variant: str):
        set_variant(variant)  # globalny wybór
        self.statusBar().showMessage(f"Model: {get_label()}")
