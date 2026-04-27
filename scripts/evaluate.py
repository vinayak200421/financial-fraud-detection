from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate teacher/student/baselines and plot ROC/PR.")
    p.add_argument(
        "--config",
        type=str,
        default="configs/eval.yaml",
        help="Path to YAML config (default: configs/eval.yaml).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode (scaffold only).",
    )
    return p.parse_args()


def main() -> None:
    _ = parse_args()
    raise NotImplementedError(
        "Evaluation not implemented yet. This CLI exists for Step 0 verification."
    )


if __name__ == "__main__":
    main()

