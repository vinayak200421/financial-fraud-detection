import logging
import os
import json
from datetime import datetime

def setup_logger(name: str = "fraud_detection") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

def create_experiment_dir(base_dir: str = "artifacts/runs") -> str:
    """Creates a unique directory for the current experiment run."""
    os.makedirs(base_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    return run_dir

def save_metrics(run_dir: str, metrics: dict, filename: str = "metrics.json"):
    """Saves metrics dictionary to a JSON file."""
    with open(os.path.join(run_dir, filename), "w") as f:
        json.dump(metrics, f, indent=4)

def save_config(run_dir: str, config: dict, filename: str = "config_snapshot.json"):
    """Saves configuration snapshot."""
    with open(os.path.join(run_dir, filename), "w") as f:
        json.dump(config, f, indent=4)
