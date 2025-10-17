from __future__ import annotations
from pathlib import Path
from PySide6 import QtWidgets
import librosa
import torch
from ..core.asr import load_model, transcribe_chunk
from ..core.constants import SR

class FilesTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        v = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.dir = QtWidgets.QLineEdit("")
        self.pick = QtWidgets.QPushButton("Wybierz…"); self.pick.clicked.connect(self._pick)
        h = QtWidgets.QHBoxLayout(); h.addWidget(self.dir, 1); h.addWidget(self.pick)
        w = QtWidgets.QWidget(); w.setLayout(h)
        form.addRow("Folder z .wav:", w)
        self.mode = QtWidgets.QComboBox(); self.mode.addItems(["transcribe", "translate"])
        self.lang = QtWidgets.QLineEdit("pl")
        form.addRow("Tryb:", self.mode)
        form.addRow("Język:", self.lang)
        v.addLayout(form)

        self.out = QtWidgets.QTextEdit(); self.out.setReadOnly(True)
        v.addWidget(self.out, 1)
        self.run = QtWidgets.QPushButton("Transkrybuj")
        v.addWidget(self.run)
        self.run.clicked.connect(self._run)

    def _pick(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder z WAV")
        if d:
            self.dir.setText(d)

    def _run(self):
        folder = Path(self.dir.text().strip())
        if not folder.is_dir():
            QtWidgets.QMessageBox.warning(self, "Błąd", "Wskaż poprawny folder z .wav")
            return
        proc, mdl, device, base = load_model(mode=self.mode.currentText(), language=self.lang.text().strip(), use_merged=True)
        wavs = sorted(folder.glob("*.wav"))
        self.out.clear()
        for w in wavs:
            audio, _ = librosa.load(str(w), sr=SR, mono=True)
            text = transcribe_chunk(audio, proc, base, device)
            self.out.append(f"[FILE] {w.name}: {text}")
