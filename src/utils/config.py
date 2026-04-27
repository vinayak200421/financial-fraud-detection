from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return yaml.safe_load(p.read_text()) or {}


def merge_overrides(cfg: Dict[str, Any], overrides: Optional[list[str]]) -> Dict[str, Any]:
    """
    Apply overrides of the form:
      key=value
      nested.key=value
    Values are parsed as YAML scalars (so 7 -> int, true -> bool, etc).
    """
    if not overrides:
        return cfg

    out = dict(cfg)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Invalid override '{ov}'. Expected key=value.")
        key, raw_val = ov.split("=", 1)
        val: Any = yaml.safe_load(raw_val)

        parts = key.split(".")
        cur: Dict[str, Any] = out
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = val
    return out

