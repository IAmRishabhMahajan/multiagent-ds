# src/utils/common.py
import yaml
from pathlib import Path
import os
from datetime import datetime


def load_config():
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("Missing config/config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def load_knobs():
    knobs_path = Path("config/knobs.yaml")
    if not knobs_path.exists():
        raise FileNotFoundError("Missing config/knobs.yaml")
    with open(knobs_path, "r") as f:
        return yaml.safe_load(f)

def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)