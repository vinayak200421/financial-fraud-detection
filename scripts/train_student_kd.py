from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train per-domain student models with knowledge distillation."
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/student_kd.yaml",
        help="Path to YAML config (default: configs/student_kd.yaml).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode (creates no training yet; scaffold only).",
    )
    return p.parse_args()


def main() -> None:
    _ = parse_args()
    raise NotImplementedError(
        "Student KD training not implemented yet. This CLI exists for Step 0 verification."
    )


if __name__ == "__main__":
    main()

