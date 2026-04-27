from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.train.metrics import classification_metrics


class _TorchWrapper(Dataset):
    def __init__(self, base: Any) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.base[idx]


def _collate_to_torch(batch: list[Dict[str, Any]]) -> Dict[str, Any]:
    X = torch.tensor(np.stack([b["X"] for b in batch], axis=0), dtype=torch.float32)
    y = torch.tensor([b["y"] for b in batch], dtype=torch.long)
    domain = [b.get("domain") for b in batch]
    return {"X": X, "y": y, "domain": domain}


@dataclass(frozen=True)
class TeacherTrainConfig:
    lr: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    weight_decay: float = 0.0
    max_train_batches: Optional[int] = None
    max_eval_batches: Optional[int] = None


@torch.no_grad()
def _predict_prob_pos(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        X = batch["X"].to(device)
        y = batch["y"].to(device)
        logits = model(X)
        prob = torch.softmax(logits, dim=-1)[:, 1]
        ys.append(y.detach().cpu().numpy())
        ps.append(prob.detach().cpu().numpy())
    return np.concatenate(ys, axis=0), np.concatenate(ps, axis=0)


def train_teacher(
    model: nn.Module,
    train_ds: Any,
    val_ds: Any,
    device: torch.device,
    cfg: TeacherTrainConfig,
) -> Dict[str, Any]:
    train_loader = DataLoader(
        _TorchWrapper(train_ds),
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_to_torch,
    )
    val_loader = DataLoader(
        _TorchWrapper(val_ds),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_to_torch,
    )

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    loss_fn = nn.CrossEntropyLoss()

    history: Dict[str, Any] = {"epochs": []}
    best = {"val_f1": -1.0, "state_dict": None, "epoch": None}

    for epoch in range(int(cfg.epochs)):
        model.train()
        running = 0.0
        seen = 0
        for bi, batch in enumerate(train_loader):
            if cfg.max_train_batches is not None and bi >= cfg.max_train_batches:
                break
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu().item()) * int(y.shape[0])
            seen += int(y.shape[0])

        train_loss = running / max(1, seen)
        yv, pv = _predict_prob_pos(model, val_loader, device=device, max_batches=cfg.max_eval_batches)
        val_metrics = classification_metrics(yv, pv) if len(yv) else {}

        row = {"epoch": epoch, "train_loss": train_loss, "val": val_metrics}
        history["epochs"].append(row)

        val_f1 = float(val_metrics.get("f1") or -1.0)
        if val_f1 > best["val_f1"]:
            best["val_f1"] = val_f1
            best["epoch"] = epoch
            best["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    history["best"] = {"val_f1": best["val_f1"], "epoch": best["epoch"]}
    history["_best_state_dict"] = best["state_dict"]
    return history

