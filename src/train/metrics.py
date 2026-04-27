from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, y_prob_pos: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob_pos = np.asarray(y_prob_pos).astype(float)
    y_pred = (y_prob_pos >= 0.5).astype(int)

    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # roc_auc requires both classes to be present.
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob_pos))
    except Exception:
        out["roc_auc"] = None

    return out

