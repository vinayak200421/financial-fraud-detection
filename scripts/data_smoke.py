from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from src.data.build import DataConfig, build_data
from src.data.preprocess import PreprocessConfig
from src.data.splits import SplitConfig
from src.utils.artifacts import create_run_dir, write_config_snapshot, write_metrics
from src.utils.config import load_yaml, merge_overrides
from src.utils.seed import SeedConfig, set_global_seed


def _default_cfg() -> Dict[str, Any]:
    return {
        "task": "data_smoke",
        "seed": 42,
        "artifacts": {"root": "artifacts", "run_id": None},
        "data": {
            "path": None,
            "label_col": "label",
            "industry_col": "industry",
            "features": None,
            "preprocess": {"impute_strategy": "median", "scale": True},
            "split": {
                "test_size": 0.3,
                "val_size_within_train": 0.1,
                "seed": 42,
                "stratify": True,
            },
        },
        "smoke": {"batch_size": 8, "num_batches": 2},
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Data pipeline smoke: load → split → preprocess → iterate batches."
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/data.yaml",
        help="Path to YAML config (default: configs/data.yaml).",
    )
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config key/value (repeatable). Example: --set data.path=./my.csv",
    )
    p.add_argument("--run-id", type=str, default=None, help="Optional run id.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = _default_cfg()
    cfg.update(load_yaml(args.config) if args.config else {})
    cfg = merge_overrides(cfg, args.overrides)
    if args.run_id:
        cfg.setdefault("artifacts", {})
        cfg["artifacts"]["run_id"] = args.run_id

    set_global_seed(SeedConfig(seed=int(cfg.get("seed", 42))))

    run_paths = create_run_dir(
        artifacts_root=cfg.get("artifacts", {}).get("root", "artifacts"),
        run_id=cfg.get("artifacts", {}).get("run_id"),
        repo_root=".",
    )
    write_config_snapshot(run_paths.config_snapshot_path, cfg)

    d = cfg["data"]
    data_cfg = DataConfig(
        path=d.get("path"),
        label_col=d.get("label_col", "label"),
        industry_col=d.get("industry_col", "industry"),
        features=d.get("features"),
        preprocess=PreprocessConfig(**(d.get("preprocess", {}) or {})),
        split=SplitConfig(**(d.get("split", {}) or {})),
    )

    built = build_data(data_cfg)

    # Iterate a couple of "batches" deterministically (no torch dependency).
    bs = int(cfg.get("smoke", {}).get("batch_size", 8))
    nb = int(cfg.get("smoke", {}).get("num_batches", 2))
    samples = []
    for i in range(min(nb * bs, len(built.train))):
        samples.append(built.train[i])

    batch_shapes = {
        "num_samples_taken": len(samples),
        "X_shape": [len(samples), int(built.train[0]["X"].shape[0])] if samples else [0, int(len(built.feature_columns))],
    }

    metrics = {
        "mode": "data_smoke",
        "fingerprint": built.fingerprint,
        "batch_shapes": batch_shapes,
    }
    write_metrics(run_paths.metrics_path, metrics)

    print(json.dumps({"run_dir": str(run_paths.run_dir)}, indent=2))


if __name__ == "__main__":
    main()

