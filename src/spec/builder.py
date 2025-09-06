# src/spec/builder.py
import json, hashlib
from pathlib import Path
import pandas as pd
from datetime import datetime

def infer_task(df: pd.DataFrame, target: str) -> str:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10:
        return "regression"
    return "classification"

def default_metric(task: str) -> str:
    return "F1" if task == "classification" else "RMSE"

def make_run_dir(artifacts_dir: Path) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = artifacts_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def build_spec(df, target, seed, artifacts_dir, override_task=None, override_metric=None):
    task = override_task or infer_task(df, target)
    metric = override_metric or default_metric(task)
    run_dir = make_run_dir(artifacts_dir)
    spec_path = run_dir / "spec.json"
    spec = {"task": task, "metric": metric, "target": target, "seed": seed}
    with open(spec_path, "w") as f: json.dump(spec, f, indent=2)
    return spec_path
