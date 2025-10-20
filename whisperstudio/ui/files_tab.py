# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple
from time import perf_counter

import librosa
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.constants import SR
from ..core.asr import load_model, transcribe_chunk
from .model_selection import get_model_args, get_label
from .async_file_picker import FileSystemPicker, AUDIO_SUFFIXES

# ───────── Metryki ─────────
_punct_keep_pl = r"\w\sąćęłńóśżź"
_rx_punct = re.compile(rf"[^{_punct_keep_pl}]+", flags=re.UNICODE)
def _norm_tokens(s: str) -> List[str]:
    s = (s or "").strip().lower()
    s = _rx_punct.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split() if s else []
def _norm_chars(s: str) -> List[str]:
    s = (s or "").strip().lower()
    s = _rx_punct.sub(" ", s)
    s = re.sub(r"\s+", "", s)
    return list(s)
def _edit_distance(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i; ai = a[i - 1]
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if ai == b[j - 1] else 1))
            prev = tmp
    return dp[-1]
def compute_wer(hyp: str, ref: str) -> Optional[float]:
    ref_toks, hyp_toks = _norm_tokens(ref), _norm_tokens(hyp)
    if not ref_toks: return None
    return 100.0 * _edit_distance(ref_toks, hyp_toks) / len(ref_toks)
def compute_cer(hyp: str, ref: str) -> Optional[float]:
    ref_ch, hyp_ch = _norm_chars(ref), _norm_chars(hyp)
    if not ref_ch: return None
    return 100.0 * _edit_distance(ref_ch, hyp_ch) / len(ref_ch)

def _default_flat_audio() -> Optional[Path]:
    p = (Path.cwd() / "flat_audio").resolve()
    return p if p.exists() and p.is_dir() else None

# ───────── Worker ─────────
class BatchWorker(QtCore.QObject):
    progress = QtCore.Signal(int)
    status = QtCore.Signal(str)
    line = QtCore.Signal(str)
    enable = QtCore.Signal(bool)
    finished = QtCore.Signal()

    def __init__(self, items: List[Tuple[Path, Optional[Path]]],
                 mode: str, language: str, model_opts: dict):
        super().__init__()
        self.items = items
        self.mode = mode
        self.language = language
        self.model_opts = model_opts
        self._stop = False

    @QtCore.Slot()
    def stop(self):
        self._stop = True

    @QtCore.Slot()
    def run(self):
        self.enable.emit(False)
        try:
            processor, model, device, base = load_model(
                variant=self.model_opts.get("variant"),
                mode=self.mode, language=self.language,
                device_str=self.model_opts.get("device", "auto"),
                base_dir=self.model_opts.get("base_dir"),
                lora_out_dir=self.model_opts.get("lora_out_dir"),
                merged_dir=self.model_opts.get("merged_dir"),
            )
            dbg = base.get("debug", {})
            sel = dbg.get("selected", {})
            msg = (
                f"Załadowano model: {self.model_opts.get('label','')} "
                f"[device={dbg.get('device','?')}]"
            )
            self.status.emit(msg)
            # wypisz skąd: wagi / processor / lora
            self.line.emit(
                "Ścieżki:\n"
                f"  • wagi:      {sel.get('model_dir','?')}\n"
                f"  • processor: {sel.get('processor_dir','?')}\n"
                f"  • LoRA:      {sel.get('lora_dir','—')}\n"
            )
        except Exception as e:
            self.status.emit(f"Błąd ładowania modelu: {e}")
            self.enable.emit(True); self.finished.emit(); return

        n = len(self.items)
        if n == 0:
            self.status.emit("Brak pozycji do transkrypcji.")
            self.enable.emit(True); self.finished.emit(); return

        sum_time = 0.0; sum_dur = 0.0
        sum_wer = 0.0; cnt_wer = 0
        sum_cer = 0.0; cnt_cer = 0

        for i, (audio_path, ref_path) in enumerate(self.items, 1):
            if self._stop: break
            try:
                audio, _sr = librosa.load(str(audio_path), sr=SR, mono=True)
                dur = float(len(audio)) / SR if len(audio) else 0.0

                t0 = perf_counter()
                hyp = transcribe_chunk(audio, processor, base, base["debug"]["device"])
                dt = perf_counter() - t0
                rtf = (dt / dur) if dur > 0.0 else 0.0
                sum_time += dt; sum_dur += max(dur, 0.0)

                msg = f"[{i:>3}/{n}] {audio_path.name} | t={dt:.2f}s"
                if dur > 0.0: msg += f" (RTF {rtf:.2f})"

                if ref_path and ref_path.is_file():
                    try:
                        ref = ref_path.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        ref = ref_path.read_text(errors="ignore")
                    wer = compute_wer(hyp, ref); cer = compute_cer(hyp, ref)
                    if wer is not None: sum_wer += wer; cnt_wer += 1; msg += f" | WER {wer:.1f}%"
                    else: msg += " | WER —"
                    if cer is not None: sum_cer += cer; cnt_cer += 1; msg += f" | CER {cer:.1f}%"
                    else: msg += " | CER —"
                else:
                    msg += " | brak TXT (pomijam ocenę)"

                self.line.emit(msg + f"\n→ {hyp}")
            except Exception as e:
                self.line.emit(f"[{i:>3}/{n}] {audio_path.name}: <błąd: {e}>")
            self.progress.emit(int(i / n * 100))

        processed = max(1, i)
        avg_time = sum_time / processed
        avg_rtf = (sum_time / sum_dur) if sum_dur > 0 else 0.0
        summary = f"\n== Podsumowanie ({processed} plików) ==\n" \
                  f"Średni czas/plik: {avg_time:.2f}s | Średnie RTF: {avg_rtf:.2f}\n"
        if cnt_wer > 0: summary += f"Średni WER: {sum_wer / cnt_wer:.2f}% (N={cnt_wer})\n"
        if cnt_cer > 0: summary += f"Średni CER: {sum_cer / cnt_cer:.2f}% (N={cnt_cer})\n"
        self.line.emit(summary)

        self.enable.emit(True); self.finished.emit()

# ───────── GUI ─────────
class FilesTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        root = QtWidgets.QVBoxLayout(self)

        # KARTA
        card = QtWidgets.QFrame(); card.setObjectName("Card")
        card.setStyleSheet("QFrame#Card { background: #0a0d14; border: 1px solid #2a2f3a; border-radius: 14px; }")
        cv = QtWidgets.QVBoxLayout(card); cv.setContentsMargins(16,16,16,16); cv.setSpacing(10)

        # Górna linia: tytuł + przyciski
        top = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Transkrypcja plików"); f = title.font(); f.setPointSize(f.pointSize()+2); f.setBold(True); title.setFont(f)
        top.addWidget(title); top.addStretch(1)
        self.btnRun = QtWidgets.QPushButton("Transkrybuj"); self.btnRun.setProperty("class", "Primary")
        self.btnStop = QtWidgets.QPushButton("STOP"); self.btnStop.setProperty("class", "Danger"); self.btnStop.setEnabled(False)
        top.addWidget(self.btnRun); top.addWidget(self.btnStop)
        cv.addLayout(top)

        form = QtWidgets.QGridLayout(); form.setHorizontalSpacing(12); form.setVerticalSpacing(8)

        row = 0
        form.addWidget(QtWidgets.QLabel("Tryb:"), row, 0)
        self.mode = QtWidgets.QComboBox(); self.mode.addItems(["transcribe", "translate"]); form.addWidget(self.mode, row, 1)
        form.addWidget(QtWidgets.QLabel("Język:"), row, 2)
        self.lang = QtWidgets.QComboBox()
        self.lang.addItems(["pl","en","de","fr","es","it","pt","ru","uk","cs","sk","hu","ro","nl","sv","no","da","fi","tr","ar","el","ja","ko","zh"])
        self.lang.setCurrentText("pl"); form.addWidget(self.lang, row, 3)

        row += 1
        form.addWidget(QtWidgets.QLabel("Wejście:"), row, 0)
        self.inputKind = QtWidgets.QComboBox(); self.inputKind.addItems(["Folder", "Plik"]); form.addWidget(self.inputKind, row, 1)

        # Folder
        row += 1
        form.addWidget(QtWidgets.QLabel("Folder audio:"), row, 0)
        self.dir = QtWidgets.QLineEdit()
        btnDir = QtWidgets.QPushButton("Wybierz…")
        w1 = QtWidgets.QWidget(); h1 = QtWidgets.QHBoxLayout(w1); h1.setContentsMargins(0,0,0,0); h1.addWidget(self.dir, 1); h1.addWidget(btnDir)
        form.addWidget(w1, row, 1, 1, 3)

        # Plik
        row += 1
        form.addWidget(QtWidgets.QLabel("Plik audio:"), row, 0)
        self.file = QtWidgets.QLineEdit()
        btnFile = QtWidgets.QPushButton("Wybierz…")
        w2 = QtWidgets.QWidget(); h2 = QtWidgets.QHBoxLayout(w2); h2.setContentsMargins(0,0,0,0); h2.addWidget(self.file, 1); h2.addWidget(btnFile)
        form.addWidget(w2, row, 1, 1, 3)

        row += 1
        self.evalChk = QtWidgets.QCheckBox("Porównaj z TXT (ten sam plik .txt obok audio)")
        form.addWidget(self.evalChk, row, 1, 1, 3)

        row += 1
        form.addWidget(QtWidgets.QLabel("Rozszerzenia (folder):"), row, 0)
        self.exts = QtWidgets.QLineEdit("wav, mp3, m4a, flac, ogg"); form.addWidget(self.exts, row, 1)

        form.addWidget(QtWidgets.QLabel("Postęp:"), row, 2)
        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0)
        form.addWidget(self.progress, row, 3)

        cv.addLayout(form)

        # Output
        self.out = QtWidgets.QTextEdit(); self.out.setReadOnly(True)
        self.out.setMinimumHeight(220)
        cv.addWidget(self.out, 1)

        root.addWidget(card, 1)

        # Sygnały
        self.inputKind.currentTextChanged.connect(self._toggle_inputs)
        btnDir.clicked.connect(self._pick_dir)
        btnFile.clicked.connect(self._pick_file)
        self.btnRun.clicked.connect(self._run)
        self.btnStop.clicked.connect(self._stop)

        self.thread: Optional[QtCore.QThread] = None
        self.worker: Optional[BatchWorker] = None

        # Domyślny folder = ./flat_audio (jeśli istnieje)
        d = _default_flat_audio()
        if d:
            self.dir.setText(str(d))

        self._toggle_inputs()

    # ── Pickery bez freeza ──
    def _pick_dir(self):
        p = FileSystemPicker.pick_dir(self)
        if p: self.dir.setText(str(p))

    def _pick_file(self):
        p = FileSystemPicker.pick_file(self, suffixes=AUDIO_SUFFIXES)
        if p: self.file.setText(str(p))

    def _toggle_inputs(self):
        is_folder = (self.inputKind.currentText() == "Folder")
        self.dir.setEnabled(is_folder)
        self.exts.setEnabled(is_folder)
        self.file.setEnabled(not is_folder)

    # ── Run/Stop ──
    def _set_enabled(self, en: bool):
        self.btnRun.setEnabled(en); self.btnStop.setEnabled(not en)
        for w in (self.mode, self.lang, self.inputKind, self.dir, self.file, self.exts, self.evalChk):
            if w is self.dir or w is self.exts:
                w.setEnabled(en and self.inputKind.currentText()=="Folder")
            elif w is self.file:
                w.setEnabled(en and self.inputKind.currentText()=="Plik")
            else:
                w.setEnabled(en)

    def _build_items(self) -> Optional[List[Tuple[Path, Optional[Path]]]]:
        items: List[Tuple[Path, Optional[Path]]] = []
        is_folder = (self.inputKind.currentText() == "Folder")
        want_eval = self.evalChk.isChecked()

        if not is_folder:
            p = Path(self.file.text().strip())
            if not p.is_file():
                QtWidgets.QMessageBox.warning(self, "Błąd", "Wskaż poprawny plik audio.")
                return None
            ref = p.with_suffix(".txt") if want_eval else None
            if want_eval and not ref.is_file():
                QtWidgets.QMessageBox.warning(self, "Brak TXT", "Nie znaleziono pliku TXT obok audio.")
                return None
            items.append((p, ref))
        else:
            folder = Path(self.dir.text().strip())
            if not folder.is_dir():
                QtWidgets.QMessageBox.warning(self, "Błąd", "Wskaż poprawny folder.")
                return None
            exts = [e.strip().lstrip(".").lower() for e in self.exts.text().split(",") if e.strip()]
            if not exts: exts = ["wav"]
            audios: List[Path] = []
            for ext in exts:
                audios.extend(sorted(folder.glob(f"*.{ext}")))
            if not audios:
                QtWidgets.QMessageBox.information(self, "Info", "Brak plików o podanych rozszerzeniach.")
                return None
            for a in audios:
                ref = a.with_suffix(".txt") if want_eval else None
                items.append((a, ref))

        return items

    def _run(self):
        items = self._build_items()
        if items is None: return

        model_opts = get_model_args(); model_opts["label"] = get_label()
        self.thread = QtCore.QThread(self)
        self.worker = BatchWorker(items, self.mode.currentText(), self.lang.currentText(), model_opts)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.status.connect(self._append_status)
        self.worker.line.connect(self._append_line)
        self.worker.enable.connect(self._set_enabled)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.out.clear(); self.progress.setValue(0)
        self.thread.start(); self._set_enabled(False)

    def _stop(self):
        if self.worker: self.worker.stop()
        if self.thread: self.thread.quit(); self.thread.wait(1500)
        self._set_enabled(True)

    # ── log ──
    def _append_status(self, s: str): self._append(f"[{s}]\n")
    def _append_line(self, s: str):   self._append(s + "\n")
    def _append(self, s: str):
        self.out.moveCursor(QtGui.QTextCursor.End)
        self.out.insertPlainText(s)
        self.out.moveCursor(QtGui.QTextCursor.End)
