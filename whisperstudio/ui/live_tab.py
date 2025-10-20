# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Optional, Tuple

import sounddevice as sd
import torch
from PySide6 import QtCore, QtGui, QtWidgets

try:
    import soundcard as sc

    HAVE_SC = True
except Exception:
    HAVE_SC = False

from ..core.asr import load_model
from ..core.streamer import LiveStreamer, StreamCfg
from .model_selection import get_label, get_model_args


class Worker(QtCore.QObject):
    delta = QtCore.Signal(str)
    status = QtCore.Signal(str)
    level = QtCore.Signal(int)
    enable = QtCore.Signal(bool)
    finished = QtCore.Signal()  # Dodany sygnał finished

    def __init__(self, cfg: StreamCfg, model_opts: dict):
        super().__init__()
        self.cfg = cfg
        self.model_opts = model_opts
        self._streamer: Optional[LiveStreamer] = None

    @QtCore.Slot()
    def stop(self):
        if self._streamer:
            self._streamer.stop()

    @QtCore.Slot()
    def run(self):
        self.enable.emit(False)
        try:
            proc, mdl, device, base = load_model(
                variant=self.model_opts.get("variant"),
                mode=self.model_opts.get("mode", "transcribe"),
                language=self.model_opts.get("language", "pl"),
                device_str=self.model_opts.get("device", "auto"),
                base_dir=self.model_opts.get("base_dir"),
                lora_out_dir=self.model_opts.get("lora_dir"),
                merged_dir=self.model_opts.get("merged_dir"),
            )
            dbg = base.get("debug", {})
            sel = dbg.get("selected", {})
            msg = (
                f"Załadowano model: {self.model_opts.get('label', '')} "
                f"[device={dbg.get('device', '?')}]"
            )
            self.status.emit(msg)
            # Wypisz ścieżki (jak w files_tab)
            self.status.emit(
                "Ścieżki:\n"
                f"  • wagi:      {sel.get('model_dir', '?')}\n"
                f"  • processor: {sel.get('processor_dir', '?')}\n"
                f"  • LoRA:      {sel.get('lora_dir', '—')}\n"
            )

        except Exception as e:
            self.status.emit(f"Błąd modelu: {e}")
            self.enable.emit(True)
            self.finished.emit()  # Zakończ, jeśli model się nie załadował
            return

        self._streamer = LiveStreamer(self.cfg)
        src = self.cfg.source
        if src == "mic":
            self.status.emit(f"Źródło: MIC (index={self.cfg.input_index})")
        else:
            self.status.emit(f"Źródło: LOOPBACK (name~='{self.cfg.loopback_name or 'auto'}')")

        try:
            self._streamer.run(proc, base, device,
                               on_delta=self.delta.emit,
                               on_status=self.status.emit,
                               on_level=self.level.emit)
        except Exception as e:
            self.status.emit(f"Błąd streamera: {e}")
        finally:
            self.enable.emit(True)
            self.finished.emit()  # Zawsze emituj 'finished' na końcu


class LiveTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        v = QtWidgets.QVBoxLayout(self)
        style = self.style()  # Pobranie stylu dla ikon

        form = QtWidgets.QFormLayout()
        self.mode = QtWidgets.QComboBox();
        self.mode.addItems(["transcribe", "translate"])
        self.lang = QtWidgets.QComboBox()
        self.lang.addItems(
            ["pl", "en", "de", "fr", "es", "it", "pt", "ru", "uk", "cs", "sk", "hu", "ro", "nl", "sv", "no", "da", "fi",
             "tr", "ar", "el", "ja", "ko", "zh"])
        self.lang.setCurrentText("pl")
        self.device = QtWidgets.QComboBox();
        self.device.addItems(["auto", "cpu", "cuda"])
        self.device.setCurrentText("cuda" if torch.cuda.is_available() else "auto")

        self.src = QtWidgets.QComboBox();
        self.src.addItems(["mic", "loopback"]);
        self.src.setCurrentText("mic")
        self.src.currentTextChanged.connect(self._toggle_device_inputs)

        # Urządzenia
        self.comboMic = QtWidgets.QComboBox()
        self.comboLoop = QtWidgets.QComboBox()
        self.btnRefresh = QtWidgets.QPushButton("Odśwież urządzenia")
        self.btnRefresh.setIcon(style.standardIcon(QtWidgets.QStyle.SP_BrowserReload))
        self.btnRefresh.clicked.connect(self._refresh_devices)

        # Pasek energii
        self.levelBar = QtWidgets.QProgressBar()
        self.levelBar.setRange(0, 100)
        self.levelBar.setFormat("Poziom: %p%")
        self.levelBar.setTextVisible(True)

        # Okna/czasy
        self.chunk = QtWidgets.QDoubleSpinBox();
        self.chunk.setRange(1.0, 30.0);
        self.chunk.setValue(6.0)
        self.stride = QtWidgets.QDoubleSpinBox();
        self.stride.setRange(0.2, 10.0);
        self.stride.setValue(2.0)

        # Rozmiar bloku — bezpieczny zakres + podpowiedź
        self.block = QtWidgets.QDoubleSpinBox()
        self.block.setRange(0.02, 0.25)  # 20–250 ms
        self.block.setSingleStep(0.01)
        self.block.setValue(0.10)  # 100 ms domyślnie
        self.block.setToolTip("Bezpieczny zakres: 0.02–0.25 s. Dla loopback przez 'soundcard' używane jest 0.05 s.")

        # VAD + autokalibracja
        self.vad = QtWidgets.QComboBox();
        self.vad.addItems(["energy", "webrtc", "off"]);
        self.vad.setCurrentText("energy")
        self.autoEnergy = QtWidgets.QCheckBox("Auto próg energii (kalibruj przy STARcie)");
        self.autoEnergy.setChecked(True)
        self.calibSec = QtWidgets.QDoubleSpinBox();
        self.calibSec.setRange(0.5, 5.0);
        self.calibSec.setSingleStep(0.5);
        self.calibSec.setValue(1.5)
        self.energy = QtWidgets.QDoubleSpinBox();
        self.energy.setDecimals(3);
        self.energy.setSingleStep(0.001);
        self.energy.setRange(0.0, 0.1);
        self.energy.setValue(0.010)

        form.addRow("Tryb:", self.mode)
        form.addRow("Język:", self.lang)
        form.addRow("Urządzenie obliczeń:", self.device)
        form.addRow("Źródło dźwięku:", self.src)
        form.addRow("Mikrofon (sounddevice):", self.comboMic)
        form.addRow("Loopback (WASAPI/soundcard):", self.comboLoop)
        form.addRow("Poziom sygnału:", self.levelBar)
        form.addRow("", self.btnRefresh)
        form.addRow("Okno [s]", self.chunk)
        form.addRow("Aktualizacja [s]", self.stride)
        form.addRow("Rozmiar bloku [s]", self.block)
        form.addRow("VAD:", self.vad)
        form.addRow("Auto-próg:", self.autoEnergy)
        form.addRow("Kalibracja [s]:", self.calibSec)
        form.addRow("Próg energii:", self.energy)

        v.addLayout(form)

        self.text = QtWidgets.QTextEdit();
        self.text.setReadOnly(True)
        v.addWidget(self.text, 1)

        h = QtWidgets.QHBoxLayout()
        self.start = QtWidgets.QPushButton("START");
        self.start.setProperty("class", "Primary")
        self.start.setIcon(style.standardIcon(QtWidgets.QStyle.SP_MediaPlay))

        self.stop = QtWidgets.QPushButton("STOP");
        self.stop.setEnabled(False);
        self.stop.setProperty("class", "Danger")
        self.stop.setIcon(style.standardIcon(QtWidgets.QStyle.SP_MediaStop))

        h.addWidget(self.start);
        h.addWidget(self.stop);
        v.addLayout(h)

        self.thread: Optional[QtCore.QThread] = None
        self.worker: Optional[Worker] = None

        self.start.clicked.connect(self._start)
        self.stop.clicked.connect(self._stop)
        self.vad.currentTextChanged.connect(self._on_vad_change)
        self.autoEnergy.stateChanged.connect(self._on_vad_change)

        self._mic_list: List[Tuple[str, int]] = []
        self._loop_list: List[str] = []

        self._refresh_devices()
        self._toggle_device_inputs()
        self._on_vad_change()

    # ---------- skanowanie urządzeń ----------
    def _refresh_devices(self):
        self._list_mic_devices()
        self._list_loopback_devices()

    def _list_mic_devices(self):
        self._mic_list.clear();
        self.comboMic.clear()
        try:
            devices = sd.query_devices()
            default_idx = sd.default.device[0]  # Pobierz domyślny indeks wejścia
            current_selection = -1

            for idx, d in enumerate(devices):
                if int(d.get("max_input_channels", 0)) > 0:
                    name = d.get("name", f"Device {idx}")
                    host = sd.query_hostapis(int(d.get("hostapi", 0))).get("name", "")
                    label = f"{name}  [{host}]"
                    self._mic_list.append((label, idx))
                    self.comboMic.addItem(label)
                    if idx == default_idx:
                        current_selection = len(self._mic_list) - 1

            if current_selection != -1:
                self.comboMic.setCurrentIndex(current_selection)  # Ustaw domyślne urz.
            elif not self._mic_list:
                self.comboMic.addItem("(brak urządzeń wejściowych)")
        except Exception as e:
            self.comboMic.addItem(f"(błąd listy: {e})")

    def _list_loopback_devices(self):
        self._loop_list.clear();
        self.comboLoop.clear()
        found_sc = False
        try:
            if HAVE_SC:
                loops = [m for m in sc.all_microphones(include_loopback=True) if getattr(m, "is_loopback", False)]
                for m in loops:
                    label = str(m.name)  # Użyj .name zamiast str(m)
                    self._loop_list.append(label)
                    self.comboLoop.addItem(label + "  [soundcard]")
                    found_sc = True
        except Exception:
            pass  # Ignoruj błędy soundcard

        try:
            devs = sd.query_devices()
            for idx, d in enumerate(devs):
                name = str(d.get("name", ""))
                if "loopback" in name.lower() and int(d.get("max_input_channels", 0)) > 0:
                    host = sd.query_hostapis(int(d.get("hostapi", 0))).get("name", "")
                    label = f"{name}  [{host}]"
                    self._loop_list.append(name)
                    self.comboLoop.addItem(label)
        except Exception:
            pass  # Ignoruj błędy sounddevice

        if not self._loop_list:
            self.comboLoop.addItem("(brak urządzeń loopback)")
        elif found_sc:
            self.comboLoop.setCurrentIndex(0)  # Preferuj soundcard jeśli jest

    def _toggle_device_inputs(self):
        is_mic = (self.src.currentText() == "mic")
        self.comboMic.setEnabled(is_mic)
        self.comboLoop.setEnabled(not is_mic)

    def _on_vad_change(self):
        is_energy = (self.vad.currentText() == "energy")
        self.autoEnergy.setEnabled(is_energy)
        self.calibSec.setEnabled(is_energy and self.autoEnergy.isChecked())
        self.energy.setEnabled(is_energy and not self.autoEnergy.isChecked())

    # ---------- konfiguracja ----------
    def _model_opts(self) -> dict:
        args = get_model_args()
        args["label"] = get_label()
        dev = self.device.currentText()
        device_str = ("cuda" if (dev == "auto" and torch.cuda.is_available()) else ("cpu" if dev == "auto" else dev))
        args.update(
            mode=self.mode.currentText(),
            language=self.lang.currentText(),
            device=device_str,
        )
        return args

    def _cfg(self) -> StreamCfg:
        cfg = StreamCfg(
            source=self.src.currentText(),
            chunk_sec=float(self.chunk.value()),
            stride_sec=float(self.stride.value()),
            block_sec=float(self.block.value()),
            vad=self.vad.currentText(),
            energy_th=float(self.energy.value()),
            auto_energy=bool(self.autoEnergy.isChecked()),
            auto_calib_sec=float(self.calibSec.value()),
        )
        if cfg.source == "mic":
            if self._mic_list and self.comboMic.currentIndex() >= 0:
                try:
                    _, idx = self._mic_list[self.comboMic.currentIndex()]
                    cfg.input_index = idx
                except IndexError:
                    cfg.input_index = None  # Fallback
            else:
                cfg.input_index = None  # Domyślne
        else:
            if self._loop_list and self.comboLoop.currentIndex() >= 0:
                try:
                    cfg.loopback_name = self._loop_list[self.comboLoop.currentIndex()]
                except IndexError:
                    cfg.loopback_name = None  # Fallback
            else:
                cfg.loopback_name = None  # Domyślne
        return cfg

    # ---------- UI helpers ----------
    def _append(self, s: str):
        self.text.moveCursor(QtGui.QTextCursor.End)
        self.text.insertPlainText(s)
        self.text.moveCursor(QtGui.QTextCursor.End)

    def _set_enabled(self, en: bool):
        self.start.setEnabled(en);
        self.stop.setEnabled(not en)
        for w in (self.mode, self.lang, self.device, self.src, self.comboMic, self.comboLoop,
                  self.chunk, self.stride, self.block, self.vad, self.energy, self.btnRefresh,
                  self.autoEnergy, self.calibSec):

            # Specjalna obsługa list urządzeń
            if w is self.comboMic:
                w.setEnabled(en and self.src.currentText() == "mic")
            elif w is self.comboLoop:
                w.setEnabled(en and self.src.currentText() == "loopback")
            # Reszta
            else:
                w.setEnabled(en)
        # Upewnij się, że logika VAD jest zastosowana po odblokowaniu
        if en:
            self._on_vad_change()

    def _start(self):
        if self.thread:  # Już działa
            return

        self.text.clear()
        self.thread = QtCore.QThread(self)
        self.worker = Worker(self._cfg(), self._model_opts())
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.delta.connect(self._append)
        self.worker.status.connect(lambda s: self._append("\n[" + s + "]\n"))
        self.worker.level.connect(self.levelBar.setValue)
        self.worker.enable.connect(self._set_enabled)

        # Sprzątanie po zakończeniu
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self._on_finish)

        self.thread.start()
        self._set_enabled(False)

    def _stop(self):
        if self.worker:
            self.worker.stop()
        # Nie czekaj tutaj, pozwól sygnałom 'finished' posprzątać
        # if self.thread:
        #     self.thread.quit(); self.thread.wait(1500)
        # self._set_enabled(True) # Zostanie wywołane przez 'enable' lub 'finished'

    def _on_finish(self):
        """Wywoływane po zakończeniu wątku, aby na pewno odblokować UI."""
        self._set_enabled(True)
        self.thread = None
        self.worker = None