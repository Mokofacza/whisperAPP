from __future__ import annotations
from pathlib import Path
from PySide6 import QtWidgets
import librosa
import torch
from ..core.asr import load_model, transcribe_chunk
from ..core.constants import SR
from ..core.model_finder import find_presets, ModelPreset


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
        self.useMerged = QtWidgets.QCheckBox("Użyj scalonego modelu (merged)");
        self.useMerged.setChecked(True)
        self.baseDir = QtWidgets.QLineEdit("whisper-small-local")
        self.loraDir = QtWidgets.QLineEdit("whisper-small-pl-lora")  # bez /lora_adapters
        self.mergedDir = QtWidgets.QLineEdit("whisper-small-pl-merged")
        form.addRow(self.useMerged)
        form.addRow("Baza:", self.baseDir)
        form.addRow("LoRA adapters:", self.loraDir)
        form.addRow("Merged dir:", self.mergedDir)

        v.addLayout(form)

        self.out = QtWidgets.QTextEdit(); self.out.setReadOnly(True)
        v.addWidget(self.out, 1)
        self.run = QtWidgets.QPushButton("Transkrybuj")
        v.addWidget(self.run)
        self.run.clicked.connect(self._run)

        # --- wybór modelu, tak jak w LIVE ---
        self.preset = QtWidgets.QComboBox()
        self.refreshBtn = QtWidgets.QPushButton("Odśwież listę")
        wrap = QtWidgets.QHBoxLayout();
        wrap.addWidget(self.preset, 1);
        wrap.addWidget(self.refreshBtn)
        wrapw = QtWidgets.QWidget();
        wrapw.setLayout(wrap)
        form.addRow("Zestaw modeli:", wrapw)

        self.useMerged = QtWidgets.QCheckBox("Użyj scalonego modelu (merged)");
        self.useMerged.setChecked(True)
        self.baseDir = QtWidgets.QLineEdit("whisper-small-local")
        self.loraDir = QtWidgets.QLineEdit("whisper-small-pl-lora/lora_adapters")
        self.mergedDir = QtWidgets.QLineEdit("whisper-small-pl-merged")
        form.addRow(self.useMerged)
        form.addRow("Baza:", self.baseDir)
        form.addRow("LoRA adapters:", self.loraDir)
        form.addRow("Merged dir:", self.mergedDir)

        self._presets: list[ModelPreset] = []
        self.refreshBtn.clicked.connect(self._refresh_models)
        self.preset.currentIndexChanged.connect(self._apply_preset)
        self._refresh_models()

    def _pick(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder z WAV")
        if d:
            self.dir.setText(d)

    def _run(self):
        folder = Path(self.dir.text().strip())
        if not folder.is_dir():
            QtWidgets.QMessageBox.warning(self, "Błąd", "Wskaż poprawny folder z .wav")
            return

        proc, mdl, device, base = load_model(
            mode=self.mode.currentText(),
            language=self.lang.text().strip(),
            use_merged=self.useMerged.isChecked(),
            base_dir=self.baseDir.text().strip(),
            lora_out_dir=self.loraDir.text().strip(),
            merged_dir=self.mergedDir.text().strip(),
        )

        wavs = sorted(folder.glob("*.wav"))
        self.out.clear()
        for w in wavs:
            audio, _ = librosa.load(str(w), sr=SR, mono=True)
            text = transcribe_chunk(audio, proc, base, device)
            self.out.append(f"[FILE] {w.name}: {text}")

    def _refresh_models(self):
        self._presets = find_presets(".")
        self.preset.clear()
        for p in self._presets:
            self.preset.addItem(p.name)
        if self._presets:
            self._apply_preset()

    def _apply_preset(self):
        if not self._presets:
            return
        p = self._presets[self.preset.currentIndex()]
        self.useMerged.setChecked(p.use_merged)
        self.baseDir.setText(str(p.base_dir))
        if p.lora_dir:
            self.loraDir.setText(str(p.lora_dir))
        if p.merged_dir:
            self.mergedDir.setText(str(p.merged_dir))

