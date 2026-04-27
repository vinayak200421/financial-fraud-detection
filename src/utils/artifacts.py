from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _utc_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def _safe_mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _try_git_commit_hash(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    checkpoints_dir: Path
    plots_dir: Path
    logs_dir: Path
    metrics_path: Path
    config_snapshot_path: Path
    run_meta_path: Path


def create_run_dir(
    artifacts_root: str | Path = "artifacts",
    run_id: Optional[str] = None,
    repo_root: str | Path = ".",
) -> RunPaths:
    artifacts_root = Path(artifacts_root)
    repo_root = Path(repo_root)
    run_id = run_id or _utc_run_id()

    run_dir = _safe_mkdir(artifacts_root / "runs" / run_id)
    checkpoints_dir = _safe_mkdir(run_dir / "checkpoints")
    plots_dir = _safe_mkdir(run_dir / "plots")
    logs_dir = _safe_mkdir(run_dir / "logs")

    metrics_path = run_dir / "metrics.json"
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    run_meta_path = run_dir / "run_meta.json"

    meta: Dict[str, Any] = {
        "run_id": run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _try_git_commit_hash(repo_root),
        "artifacts_root": str(artifacts_root.resolve()),
    }
    run_meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    return RunPaths(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        plots_dir=plots_dir,
        logs_dir=logs_dir,
        metrics_path=metrics_path,
        config_snapshot_path=config_snapshot_path,
        run_meta_path=run_meta_path,
    )


def write_config_snapshot(path: Path, cfg: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def write_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2) + "\n")

