from __future__ import annotations
from PySide6 import QtWidgets
from .live_tab import LiveTab
from .files_tab import FilesTab
from .train_tab import TrainTab
from .merge_tab import MergeTab

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
