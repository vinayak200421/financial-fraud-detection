from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from src.utils.artifacts import create_run_dir, write_config_snapshot, write_metrics
from src.utils.config import load_yaml, merge_overrides
from src.utils.device import get_device
from src.utils.seed import SeedConfig, set_global_seed


def _default_cfg() -> Dict[str, Any]:
    return {
        "task": "train_teacher",
        "seed": 42,
        "device": {"prefer_cuda": True},
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
        "model": {
            "d_model": 64,
            "nhead": 6,
            "num_layers": 2,
            "dim_feedforward": 1024,
            "dropout": 0.2,
            "pooling": "mean",
        },
        "train": {
            "lr": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "weight_decay": 0.0,
            "max_train_batches": None,
            "max_eval_batches": None,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the teacher Transformer fraud model.")
    p.add_argument(
        "--config",
        type=str,
        default="configs/teacher.yaml",
        help="Path to YAML config (default: configs/teacher.yaml).",
    )
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config key/value (repeatable). Example: --set train.epochs=2",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id (default: auto timestamp).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: no training; only creates artifacts + dumps config.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = _default_cfg()
    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg.update(load_yaml(cfg_path))

    cfg = merge_overrides(cfg, args.overrides)
    if args.run_id:
        cfg.setdefault("artifacts", {})
        cfg["artifacts"]["run_id"] = args.run_id

    set_global_seed(SeedConfig(seed=int(cfg.get("seed", 42))))
    device = get_device(bool(cfg.get("device", {}).get("prefer_cuda", True)))

    run_paths = create_run_dir(
        artifacts_root=cfg.get("artifacts", {}).get("root", "artifacts"),
        run_id=cfg.get("artifacts", {}).get("run_id"),
        repo_root=".",
    )
    write_config_snapshot(run_paths.config_snapshot_path, cfg)

    if args.smoke:
        write_metrics(
            run_paths.metrics_path,
            {
                "mode": "smoke",
                "device": device,
                "note": "Scaffold verified.",
            },
        )
        print(json.dumps({"run_dir": str(run_paths.run_dir)}, indent=2))
        return

    # Heavy deps are imported lazily so `--help` works even without torch installed.
    import torch

    from src.data.build import DataConfig, build_data
    from src.data.preprocess import PreprocessConfig
    from src.data.splits import SplitConfig
    from src.models.transformer_fraud import TransformerFraudClassifier
    from src.train.teacher import TeacherTrainConfig, train_teacher

    d = cfg["data"]
    if not d.get("path"):
        raise ValueError("data.path is required for training. Set it in configs/teacher.yaml or via --set.")

    data_cfg = DataConfig(
        path=d.get("path"),
        label_col=d.get("label_col", "label"),
        industry_col=d.get("industry_col", "industry"),
        features=d.get("features"),
        preprocess=PreprocessConfig(**(d.get("preprocess", {}) or {})),
        split=SplitConfig(**(d.get("split", {}) or {})),
    )
    built = build_data(data_cfg)

    m = cfg["model"]
    model = TransformerFraudClassifier(
        num_features=len(built.feature_columns),
        d_model=int(m.get("d_model", 64)),
        nhead=int(m.get("nhead", 6)),
        num_layers=int(m.get("num_layers", 2)),
        dim_feedforward=int(m.get("dim_feedforward", 1024)),
        dropout=float(m.get("dropout", 0.2)),
        pooling=str(m.get("pooling", "mean")),
        num_classes=2,
    )

    t = cfg["train"]
    train_cfg = TeacherTrainConfig(
        lr=float(t.get("lr", 0.001)),
        batch_size=int(t.get("batch_size", 32)),
        epochs=int(t.get("epochs", 100)),
        weight_decay=float(t.get("weight_decay", 0.0)),
        max_train_batches=t.get("max_train_batches"),
        max_eval_batches=t.get("max_eval_batches"),
    )

    torch_device = torch.device(device)
    history = train_teacher(
        model=model,
        train_ds=built.train,
        val_ds=built.val,
        device=torch_device,
        cfg=train_cfg,
    )

    # Save best checkpoint if we found one.
    best_sd = history.pop("_best_state_dict", None)
    if best_sd is not None:
        ckpt_path = run_paths.checkpoints_dir / "teacher_best.pt"
        torch.save(
            {
                "state_dict": best_sd,
                "feature_columns": built.feature_columns,
                "model_cfg": m,
                "data_cfg": d,
            },
            ckpt_path,
        )

    write_metrics(
        run_paths.metrics_path,
        {
            "mode": "train_teacher",
            "device": device,
            "data_fingerprint": built.fingerprint,
            "history": history,
            "checkpoint": "checkpoints/teacher_best.pt" if best_sd is not None else None,
        },
    )
    print(json.dumps({"run_dir": str(run_paths.run_dir)}, indent=2))


if __name__ == "__main__":
    main()

