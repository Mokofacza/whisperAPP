from __future__ import annotations
from PySide6 import QtWidgets, QtCore
from ..training.lora_trainer import run_lora
from ..training.full_trainer import run_full


class TrainTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        v = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.data = QtWidgets.QLineEdit("flat_audio")
        self.model = QtWidgets.QLineEdit("openai/whisper-small")
        self.out_lora = QtWidgets.QLineEdit("whisper-small-pl-lora")
        self.out_full = QtWidgets.QLineEdit("whisper-small-pl-fullft")
        self.mode = QtWidgets.QComboBox();
        self.mode.addItems(["LoRA", "Full FT"]);
        self.mode.setCurrentText("LoRA")
        self.epochs = QtWidgets.QSpinBox();
        self.epochs.setRange(1, 50);
        self.epochs.setValue(10)
        self.batch = QtWidgets.QSpinBox();
        self.batch.setRange(1, 64);
        self.batch.setValue(8)

        form.addRow("Dane (wav/txt):", self.data)
        form.addRow("Model bazowy:", self.model)
        form.addRow("Wyjście LoRA:", self.out_lora)
        form.addRow("Wyjście FullFT:", self.out_full)
        form.addRow("Tryb treningu:", self.mode)
        form.addRow("Epoki:", self.epochs)
        form.addRow("Batch size:", self.batch)
        v.addLayout(form)

        self.log = QtWidgets.QTextEdit();
        self.log.setReadOnly(True)
        v.addWidget(self.log, 1)

        self.start = QtWidgets.QPushButton("Start treningu")
        self.start.setProperty("class", "Primary")
        self.start.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))

        v.addWidget(self.start)
        self.start.clicked.connect(self._start)

        self.proc = None
        self.timer = QtCore.QTimer(self);
        self.timer.timeout.connect(self._pump)

        self.mode.currentTextChanged.connect(self._toggle_outputs)
        self._toggle_outputs()

    def _toggle_outputs(self):
        is_lora = (self.mode.currentText() == "LoRA")
        self.out_lora.setEnabled(is_lora)
        self.out_full.setEnabled(not is_lora)

    def _append(self, s: str):
        self.log.moveCursor(self.log.textCursor().End)
        self.log.insertPlainText(s)
        self.log.moveCursor(self.log.textCursor().End)

    def _set_enabled(self, en: bool):
        self.start.setEnabled(en)
        for w in (self.data, self.model, self.out_lora, self.out_full, self.mode, self.epochs, self.batch):
            w.setEnabled(en)
        if en:  # Zastosuj logikę włączania/wyłączania
            self._toggle_outputs()

    def _start(self):
        if self.proc is not None:
            return

        self._set_enabled(False)
        self.log.clear()
        self._append("Rozpoczynanie treningu...\n")

        try:
            if self.mode.currentText() == "LoRA":
                self.proc = run_lora(
                    self.data.text(),
                    self.model.text(),
                    self.out_lora.text(),
                    epochs=int(self.epochs.value()),
                    batch_size=int(self.batch.value()),
                )
            else:
                self.proc = run_full(
                    self.data.text(),
                    self.model.text(),
                    self.out_full.text(),
                    epochs=int(self.epochs.value()),
                    batch_size=int(self.batch.value()),
                )
            self.timer.start(200)
        except Exception as e:
            self._append(f"\n[BŁĄD KRYTYCZNY PRZED STARTEM]\n{e}\n")
            self._set_enabled(True)

    def _pump(self):
        if self.proc is None:
            return

        try:
            line = self.proc.stdout.readline()
        except Exception:  # np. błąd dekodowania
            line = ""

        if line:
            self._append(line)
        elif self.proc.poll() is not None:
            # Proces zakończony
            self._append(f"\n[ZAKOŃCZONE (kod: {self.proc.poll()})]\n")
            self.timer.stop()
            self.proc = None
            self._set_enabled(True)