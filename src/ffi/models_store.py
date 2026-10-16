"""Versioned model artifacts shipped in src/ffi/models/ (BLUEPRINT §10.3). Runs only load them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODELS_DIR = Path(__file__).resolve().parent / "models"


def load_model(name: str) -> tuple[Any, dict] | None:
    """Return (lightgbm.Booster, metadata) or None when the artifact has not been trained."""
    model_path = MODELS_DIR / f"{name}.lgb.txt"
    meta_path = MODELS_DIR / f"{name}.json"
    if not model_path.exists() or not meta_path.exists():
        return None
    import lightgbm as lgb

    return lgb.Booster(model_file=str(model_path)), json.loads(meta_path.read_text(encoding="utf-8"))


def save_model(name: str, booster: Any, meta: dict) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODELS_DIR / f"{name}.lgb.txt"))
    (MODELS_DIR / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return MODELS_DIR / f"{name}.lgb.txt"


def load_calibration() -> dict[str, list[list[float]]]:
    """Isotonic calibration maps per detector: {detector_id: [[severity, probability], ...]}."""
    p = MODELS_DIR / "calibration.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))
