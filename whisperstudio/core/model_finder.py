# whisperstudio/core/model_finder.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass
class ModelPreset:
    name: str
    use_merged: bool
    base_dir: Path
    lora_dir: Optional[Path] = None
    merged_dir: Optional[Path] = None

def _is_hf_model_dir(d: Path) -> bool:
    return (
        (d / "config.json").exists()
        or (d / "generation_config.json").exists()
        or (d / "pytorch_model.bin").exists()
        or any(d.glob("*.safetensors"))
    )

def find_presets(root: Path | str = ".") -> List[ModelPreset]:
    root = Path(root).resolve()
    dirs = [p for p in root.iterdir() if p.is_dir()]

    # Katalogi wyglądające na „zwykły” model HF (to może być baza albo już scalony merged)
    hf_dirs = [d for d in dirs if _is_hf_model_dir(d)]
    # Katalogi z LoRA (po prostu szukamy podkatalogu 'lora_adapters')
    lora_roots = [d for d in dirs if (d / "lora_adapters").exists()]

    presets: List[ModelPreset] = []

    # Każdy katalog HF wystaw jako preset merged/HF (działa dla Twojego 'whisper-small-pl-merged' i 'whisper-small-local')
    for d in hf_dirs:
        presets.append(ModelPreset(
            name=f"{d.name} (HF/merged)",
            use_merged=True,
            base_dir=d,
            merged_dir=d
        ))

    # Wybierz sensowną bazę dla LoRA (preferuj tę z nazwą 'local' lub 'small')
    base_default = None
    for cand in hf_dirs:
        n = cand.name.lower()
        if "local" in n or "small" in n or "base" in n:
            base_default = cand
            break
    if base_default is None and hf_dirs:
        base_default = hf_dirs[0]

    # Presety LoRA: każda „loRa-owa” paczka + domyślna baza (u Ciebie: whisper-small-local)
    for lroot in lora_roots:
        presets.append(ModelPreset(
            name=f"{lroot.name} (LoRA)",
            use_merged=False,
            base_dir=base_default if base_default else lroot,
            lora_dir=lroot / "lora_adapters",
        ))

    # Usuń duplikaty po nazwie, zachowaj kolejność
    seen, out = set(), []
    for p in presets:
        if p.name in seen:
            continue
        out.append(p); seen.add(p.name)
    return out
