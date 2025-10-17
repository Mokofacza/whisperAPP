from __future__ import annotations
from PySide6 import QtWidgets
from ..core.model_finder import find_presets
from .live_tab import LiveTab
from .files_tab import FilesTab
from .train_tab import TrainTab
from .merge_tab import MergeTab

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper Studio")
        self.resize(1120, 760)

        self.model_selector = QtWidgets.QComboBox()

        presets = find_presets("./")
        self.model_selector.addItems([preset.base_dir.name for preset in presets])
        self.model_selector.currentIndexChanged.connect(self._on_model_selected)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(LiveTab(), "LIVE")
        tabs.addTab(FilesTab(), "Pliki")
        tabs.addTab(TrainTab(), "Trening")
        tabs.addTab(MergeTab(), "Merge/Export")

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(self.model_selector)
        main_layout.addWidget(tabs)

        main_widget = QtWidgets.QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def _on_model_selected(self, index):
        selected_model = self.model_selector.itemText(index)
        print(f"Selected model: {selected_model}")
