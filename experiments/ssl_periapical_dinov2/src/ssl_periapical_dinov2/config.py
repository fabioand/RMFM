from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config nao encontrada: {cfg_path}")

    suffix = cfg_path.suffix.lower()
    raw = cfg_path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(raw)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("PyYAML nao disponivel para ler arquivo .yaml/.yml") from exc
        out = yaml.safe_load(raw)
        if not isinstance(out, dict):
            raise ValueError(f"Config invalida em {cfg_path}: esperado objeto no topo")
        return out
    raise ValueError(f"Formato de config nao suportado: {cfg_path}")


def save_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

