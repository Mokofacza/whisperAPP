from __future__ import annotations
from PySide6 import QtWidgets
from ..core.merge import merge_lora

class MergeTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        v = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.base = QtWidgets.QLineEdit("whisper-small-local")
        self.adapters = QtWidgets.QLineEdit("whisper-small-pl-lora/lora_adapters")
        self.out = QtWidgets.QLineEdit("whisper-small-pl-merged")
        form.addRow("Baza (model):", self.base)
        form.addRow("Adaptery LoRA:", self.adapters)
        form.addRow("Wyjście (merged):", self.out)
        v.addLayout(form)

        self.btn = QtWidgets.QPushButton("Scal i zapisz")
        self.log = QtWidgets.QTextEdit(); self.log.setReadOnly(True)
        v.addWidget(self.btn); v.addWidget(self.log, 1)
        self.btn.clicked.connect(self._run)

    def _run(self):
        try:
            out = merge_lora(self.base.text().strip(), self.adapters.text().strip(), self.out.text().strip())
            self.log.append(f"Zapisano scalony model do: {out}")
        except Exception as e:
            self.log.append(str(e))
