from __future__ import annotations
from typing import Optional
from PySide6 import QtCore, QtGui, QtWidgets
import torch

from ..core.asr import load_model
from ..core.streamer import LiveStreamer, StreamCfg
from ..core.model_finder import find_presets, ModelPreset


class Worker(QtCore.QObject):
    delta = QtCore.Signal(str)
    status = QtCore.Signal(str)
    enable = QtCore.Signal(bool)

    def __init__(self, cfg: StreamCfg, model_opts: dict):
        super().__init__()
        self.cfg = cfg
        self.model_opts = model_opts
        self._stop = False

    @QtCore.Slot()
    def stop(self):
        self._stop = True

    @QtCore.Slot()
    def run(self):
        self.enable.emit(False)
        try:
            proc, mdl, device, base = load_model(
                mode=self.model_opts.get("mode", "transcribe"),
                language=self.model_opts.get("language", "pl"),
                device_str=self.model_opts.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
                use_merged=self.model_opts.get("use_merged", True),
                base_dir=self.model_opts.get("base_dir"),
                lora_out_dir=self.model_opts.get("lora_dir"),
                merged_dir=self.model_opts.get("merged_dir"),
            )
        except Exception as e:
            self.status.emit(f"Błąd modelu: {e}")
            self.enable.emit(True)
            return

        streamer = LiveStreamer(self.cfg)

        def on_delta(s: str):
            self.delta.emit(s)

        def on_status(s: str):
            self.status.emit(s)

        streamer.run(proc, base, device, on_delta, on_status)
        self.enable.emit(True)

class LiveTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        v = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.preset = QtWidgets.QComboBox()
        self.refreshBtn = QtWidgets.QPushButton("Odśwież listę")
        wrap = QtWidgets.QHBoxLayout();
        wrap.addWidget(self.preset, 1);
        wrap.addWidget(self.refreshBtn)
        wrapw = QtWidgets.QWidget();
        wrapw.setLayout(wrap)
        form.addRow("Zestaw modeli:", wrapw)

        self._presets: list[ModelPreset] = []
        self.refreshBtn.clicked.connect(self._refresh_models)
        self.preset.currentIndexChanged.connect(self._apply_preset)

        self.mode = QtWidgets.QComboBox(); self.mode.addItems(["transcribe", "translate"])
        self.lang = QtWidgets.QComboBox()
        self.lang.addItems(["pl","en","de","fr","es","it","pt","ru","uk","cs","sk","hu","ro","nl","sv","no","da","fi","tr","ar","el","ja","ko","zh"])
        self.lang.setCurrentText("pl")
        self.device = QtWidgets.QComboBox(); self.device.addItems(["auto", "cpu", "cuda"])
        self.device.setCurrentText("cuda" if torch.cuda.is_available() else "auto")
        self.src = QtWidgets.QComboBox(); self.src.addItems(["mic", "loopback"]); self.src.setCurrentText("mic")
        self.chunk = QtWidgets.QDoubleSpinBox(); self.chunk.setRange(1.0, 30.0); self.chunk.setValue(6.0)
        self.stride = QtWidgets.QDoubleSpinBox(); self.stride.setRange(0.2, 10.0); self.stride.setValue(1.5)
        self.block = QtWidgets.QDoubleSpinBox(); self.block.setRange(0.0, 1.0); self.block.setValue(0.2)
        self.vad = QtWidgets.QComboBox(); self.vad.addItems(["energy", "webrtc", "off"]); self.vad.setCurrentText("energy")
        self.energy = QtWidgets.QDoubleSpinBox(); self.energy.setDecimals(3); self.energy.setSingleStep(0.001); self.energy.setRange(0.0, 0.1); self.energy.setValue(0.008)
        form.addRow("Tryb:", self.mode)
        form.addRow("Język:", self.lang)
        form.addRow("Urządzenie obliczeń:", self.device)
        form.addRow("Źródło dźwięku:", self.src)
        form.addRow("Okno [s]", self.chunk)
        form.addRow("Aktualizacja [s]", self.stride)
        form.addRow("Rozmiar bloku [s]", self.block)
        form.addRow("VAD:", self.vad)
        form.addRow("Próg energii:", self.energy)

        self.useMerged = QtWidgets.QCheckBox("Użyj scalonego modelu (merged)"); self.useMerged.setChecked(True)
        self.baseDir = QtWidgets.QLineEdit("whisper-small-local")
        self.loraDir = QtWidgets.QLineEdit("whisper-small-pl-lora")
        self.mergedDir = QtWidgets.QLineEdit("whisper-small-pl-merged")
        form.addRow(self.useMerged)
        form.addRow("Baza:", self.baseDir)
        form.addRow("LoRA adapters:", self.loraDir)
        form.addRow("Merged dir:", self.mergedDir)

        v.addLayout(form)

        self.text = QtWidgets.QTextEdit(); self.text.setReadOnly(True)
        v.addWidget(self.text, 1)

        h = QtWidgets.QHBoxLayout()
        self.start = QtWidgets.QPushButton("START"); self.stop = QtWidgets.QPushButton("STOP"); self.stop.setEnabled(False)
        h.addWidget(self.start); h.addWidget(self.stop); v.addLayout(h)

        self.thread: Optional[QtCore.QThread] = None
        self.worker: Optional[Worker] = None

        self.start.clicked.connect(self._start)
        self.stop.clicked.connect(self._stop)

        self._refresh_models()

    def _model_opts(self) -> dict:
        dev = self.device.currentText()
        device_str = ("cuda" if (dev == "auto" and torch.cuda.is_available()) else ("cpu" if dev == "auto" else dev))
        return dict(
            mode=self.mode.currentText(), language=self.lang.currentText(), device=device_str,
            use_merged=self.useMerged.isChecked(), base_dir=self.baseDir.text().strip(),
            lora_dir=self.loraDir.text().strip(), merged_dir=self.mergedDir.text().strip(),
        )

    def _cfg(self) -> StreamCfg:
        return StreamCfg(
            source=self.src.currentText(),
            chunk_sec=float(self.chunk.value()),
            stride_sec=float(self.stride.value()),
            block_sec=float(self.block.value()),
            vad=self.vad.currentText(),
            energy_th=float(self.energy.value()),
        )

    def _append(self, s: str):
        self.text.moveCursor(QtGui.QTextCursor.End)
        self.text.insertPlainText(s)
        self.text.moveCursor(QtGui.QTextCursor.End)

    def _set_enabled(self, en: bool):
        self.start.setEnabled(en); self.stop.setEnabled(not en)

    def _start(self):
        self.text.clear()
        self.thread = QtCore.QThread(self)
        self.worker = Worker(self._cfg(), self._model_opts())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.delta.connect(self._append)
        self.worker.status.connect(lambda s: self._append("\n[" + s + "]\n"))
        self.worker.enable.connect(self._set_enabled)
        self.thread.start()
        self._set_enabled(False)

    def _stop(self):
        if self.worker:
            self.worker.stop()
        if self.thread:
            self.thread.quit(); self.thread.wait(1500)
        self._set_enabled(True)

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

