from __future__ import annotations
from PySide6 import QtWidgets, QtCore
from ..core.merge import merge_lora


class MergeWorker(QtCore.QObject):
    """Worker do scalania w osobnym wątku, by nie blokować GUI"""
    finished = QtCore.Signal(str)  # Sygnał (wynik_lub_błąd)

    def __init__(self, base: str, adapters: str, out: str):
        super().__init__()
        self.base = base
        self.adapters = adapters
        self.out = out

    @QtCore.Slot()
    def run(self):
        try:
            out_path = merge_lora(self.base, self.adapters, self.out)
            self.finished.emit(f"Zapisano scalony model do: {out_path}")
        except Exception as e:
            self.finished.emit(f"[BŁĄD]\n{e}")


class MergeTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        v = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.base = QtWidgets.QLineEdit("openai/whisper-small")
        self.adapters = QtWidgets.QLineEdit("whisper-small-pl-lora/lora_adapters")
        self.out = QtWidgets.QLineEdit("whisper-small-pl-merged")
        form.addRow("Baza (model):", self.base)
        form.addRow("Adaptery LoRA:", self.adapters)
        form.addRow("Wyjście (merged):", self.out)
        v.addLayout(form)

        self.btn = QtWidgets.QPushButton("Scal i zapisz")
        self.btn.setProperty("class", "Primary")
        self.btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))

        self.log = QtWidgets.QTextEdit();
        self.log.setReadOnly(True)
        v.addWidget(self.btn);
        v.addWidget(self.log, 1)
        self.btn.clicked.connect(self._run)

        self.thread: Optional[QtCore.QThread] = None
        self.worker: Optional[MergeWorker] = None

    def _set_enabled(self, en: bool):
        self.btn.setEnabled(en)
        self.base.setEnabled(en)
        self.adapters.setEnabled(en)
        self.out.setEnabled(en)

    def _run(self):
        if self.thread:  # Już działa
            return

        self._set_enabled(False)
        self.log.clear()
        self.log.append("Rozpoczynam scalanie...")

        self.thread = QtCore.QThread(self)
        self.worker = MergeWorker(
            self.base.text().strip(),
            self.adapters.text().strip(),
            self.out.text().strip()
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_finish)

        # Sprzątanie
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def _on_finish(self, result: str):
        self.log.append(result)
        self._set_enabled(True)
        self.thread = None
        self.worker = None