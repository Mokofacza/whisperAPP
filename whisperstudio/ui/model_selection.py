# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict
from PySide6 import QtCore, QtWidgets

# Globalny, prosty stan
_CURRENT: Dict[str, Optional[str]] = {
    "variant": "small-local",   # small-local | small-lora | small-merged | small-full
    "label":   "Whisper Small — Local",
    "base_dir": None,
    "lora_dir": None,
    "merged_dir": None,
}

def _maybe(path: Path) -> Optional[str]:
    try:
        return str(path) if path.exists() else None
    except Exception:
        return None

def _autofill_paths(variant: str) -> None:
    """
    Dopasowane do Twojej struktury katalogów ze screena:
      whisper-small-local
      whisper-small-pl-lora
      whisper-small-pl-merged
      whisper-small-pl-fullft
    (wszystko w katalogu projektu)
    """
    root = Path.cwd()
    if variant == "small-local":
        _CURRENT["base_dir"]   = _maybe(root / "whisper-small-local")
        _CURRENT["lora_dir"]   = None
        _CURRENT["merged_dir"] = None
        _CURRENT["label"]      = "Whisper Small — Local"

    elif variant == "small-lora":
        _CURRENT["base_dir"]   = _maybe(root / "whisper-small-local")
        _CURRENT["lora_dir"]   = _maybe(root / "whisper-small-pl-lora")
        _CURRENT["merged_dir"] = None
        _CURRENT["label"]      = "Whisper Small — LoRA"

    elif variant == "small-merged":
        _CURRENT["base_dir"]   = None
        _CURRENT["lora_dir"]   = None
        _CURRENT["merged_dir"] = _maybe(root / "whisper-small-pl-merged")
        _CURRENT["label"]      = "Whisper Small — Merged"

    else:  # small-full
        _CURRENT["base_dir"]   = _maybe(root / "whisper-small-pl-fullft")
        _CURRENT["lora_dir"]   = None
        _CURRENT["merged_dir"] = None
        _CURRENT["label"]      = "Whisper Small — Full (fine-tuned)"

def get_model_args() -> dict:
    return {
        "variant":   _CURRENT["variant"],
        "base_dir":  _CURRENT["base_dir"],
        "lora_out_dir": _CURRENT["lora_dir"],
        "merged_dir":_CURRENT["merged_dir"],
    }

def get_label() -> str:
    return str(_CURRENT.get("label", ""))

class ModelSelector(QtWidgets.QGroupBox):
    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__("Model")
        layout = QtWidgets.QHBoxLayout(self)
        self.combo = QtWidgets.QComboBox()
        self.combo.addItem("Whisper Small — Local",  "small-local")
        self.combo.addItem("Whisper Small — LoRA",   "small-lora")
        self.combo.addItem("Whisper Small — Merged", "small-merged")
        self.combo.addItem("Whisper Small — Full",   "small-full")
        self.combo.currentIndexChanged.connect(self._on_change)
        layout.addWidget(self.combo)
        # startowo: local
        self.combo.setCurrentIndex(0)
        self._on_change() # Wywołaj ręcznie, aby zainicjować ścieżki

    @QtCore.Slot()
    def _on_change(self):
        variant = self.combo.currentData()
        _CURRENT["variant"] = variant
        _autofill_paths(variant)
        self.changed.emit()