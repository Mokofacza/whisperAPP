# -*- coding: utf-8 -*-
from __future__ import annotations
from PySide6 import QtCore, QtGui, QtWidgets

from .theme import apply_theme
from .live_tab import LiveTab
from .files_tab import FilesTab
from .model_selection import ModelSelector
try:
    from .train_tab import TrainTab
except Exception:
    TrainTab = None
try:
    from .merge_tab import MergeTab
except Exception:
    MergeTab = None


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper Studio")
        self.resize(1200, 800)

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        h = QtWidgets.QHBoxLayout(central); h.setContentsMargins(10,10,10,10); h.setSpacing(10)

        # Sidebar
        side = QtWidgets.QFrame(); side.setObjectName("Sidebar"); side.setFixedWidth(260)
        sv = QtWidgets.QVBoxLayout(side); sv.setContentsMargins(12,12,12,12); sv.setSpacing(10)

        logo = QtWidgets.QLabel("Whisper Studio")
        f = logo.font(); f.setPointSize(f.pointSize()+3); f.setBold(True); logo.setFont(f)
        sv.addWidget(logo)

        # ► WYBÓR MODELU (globalny)
        self.modelSelector = ModelSelector()
        sv.addWidget(self.modelSelector)

        sv.addSpacing(6)
        self.btnLive  = self._side_button("🎤  LIVE")
        self.btnFiles = self._side_button("📁  PLIKI")
        self.btnTrain = self._side_button("🧪  TRENING", enabled=(TrainTab is not None))
        self.btnMerge = self._side_button("🧬  MERGE",  enabled=(MergeTab is not None))
        for b in (self.btnLive, self.btnFiles, self.btnTrain, self.btnMerge):
            sv.addWidget(b)
        sv.addStretch(1)

        # Content
        self.stack = QtWidgets.QStackedWidget()
        self.liveTab  = LiveTab()
        self.filesTab = FilesTab()
        self.trainTab = TrainTab() if TrainTab else QtWidgets.QWidget()
        self.mergeTab = MergeTab() if MergeTab else QtWidgets.QWidget()

        self.stack.addWidget(self.liveTab)   # 0
        self.stack.addWidget(self.filesTab)  # 1
        self.stack.addWidget(self.trainTab)  # 2
        self.stack.addWidget(self.mergeTab)  # 3

        self.btnLive.clicked.connect(lambda: self._switch(0, self.btnLive))
        self.btnFiles.clicked.connect(lambda: self._switch(1, self.btnFiles))
        self.btnTrain.clicked.connect(lambda: self._switch(2, self.btnTrain))
        self.btnMerge.clicked.connect(lambda: self._switch(3, self.btnMerge))
        self._switch(0, self.btnLive)

        h.addWidget(side)
        h.addWidget(self.stack, 1)

        self.setStatusBar(QtWidgets.QStatusBar())

    def _side_button(self, text: str, enabled: bool = True) -> QtWidgets.QPushButton:
        b = QtWidgets.QPushButton(text); b.setObjectName("SideItem")
        b.setCheckable(True); b.setEnabled(enabled)
        return b

    def _switch(self, idx: int, active_btn: QtWidgets.QPushButton):
        self.stack.setCurrentIndex(idx)
        for btn in (self.btnLive, self.btnFiles, self.btnTrain, self.btnMerge):
            btn.setChecked(btn is active_btn)
            btn.setProperty("active", btn is active_btn)
            btn.style().unpolish(btn); btn.style().polish(btn)


def launch():
    app = QtWidgets.QApplication([])
    apply_theme(app)
    w = MainWindow(); w.show()
    app.exec()
