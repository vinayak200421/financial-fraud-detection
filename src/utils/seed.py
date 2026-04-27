from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedConfig:
    seed: int = 42
    deterministic_torch: bool = True
    cuda_benchmark: bool = False


def set_global_seed(cfg: SeedConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)

    try:
        import torch

        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        if cfg.deterministic_torch:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = bool(cfg.cuda_benchmark)
        else:
            torch.backends.cudnn.benchmark = bool(cfg.cuda_benchmark)
    except Exception:
        # Torch is optional at import-time for some tasks (e.g. config inspection).
        return

