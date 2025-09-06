# src/cleaner/cleaner.py
import pandas as pd
import numpy as np
import json
from pathlib import Path

def clean_dataframe(df: pd.DataFrame, target: str, run_dir: Path, knobs: dict) -> tuple[Path, Path]:
    """
    Clean dataframe using knobs, guided by existing data_card.json.
    Returns (cleaned_data_path, cleaning_plan_path).
    """
    plan = {"dropped_columns": [], "imputations": {}, "actions": []}

    # Load data card for column stats
    data_card_path = run_dir / "data_card.json"
    if not data_card_path.exists():
        raise FileNotFoundError("Missing data_card.json. Run profiler first.")
    with open(data_card_path, "r") as f:
        data_card = json.load(f)

    # Fail if too much target missing
    tgt_info = next(c for c in data_card["columns"] if c["name"] == target)
    if tgt_info["missing_pct"] > knobs["target_missing_fail"]:
        raise ValueError(f"Target {target} has {tgt_info['missing_pct']:.2%} missing values")

    # Drop high-missing columns
    for col in data_card["columns"]:
        if col["name"] == target:
            continue
        if col["missing_pct"] >= knobs["drop_missing_threshold"]:
            df = df.drop(columns=[col["name"]])
            plan["dropped_columns"].append(col["name"])

    # Trim strings
    if knobs.get("trim_strings", True):
        for col in df.select_dtypes(include=["object", "string"]).columns:
            df[col] = df[col].astype(str).str.strip()
        plan["actions"].append("trim_strings")

    # Normalize booleans
    if knobs.get("normalize_booleans", True):
        for col in df.select_dtypes(include=["object", "bool"]).columns:
            if set(df[col].dropna().unique()).issubset({"0","1","True","False","true","false"}):
                df[col] = df[col].astype(str).str.lower().map(
                    {"true": 1, "false": 0, "1": 1, "0": 0}
                )
        plan["actions"].append("normalize_booleans")

    # Imputation (use knobs only, but log into plan)
    for col in df.columns:
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            strategy = knobs["impute_numeric"]
            fill_val = df[col].median() if strategy == "median" else df[col].mean()
            df[col] = df[col].fillna(fill_val)
            plan["imputations"][col] = {
                "type": "numeric",
                "strategy": strategy,
                "value": float(fill_val),
            }
        else:
            strategy = knobs["impute_categorical"]
            if strategy == "most_frequent":
                fill_val = df[col].mode(dropna=True)[0] if not df[col].mode().empty else "missing"
            else:
                fill_val = "missing"
            df[col] = df[col].fillna(fill_val)
            plan["imputations"][col] = {
                "type": "categorical",
                "strategy": strategy,
                "value": str(fill_val),
            }

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        plan["actions"].append(f"dropped_duplicates ({before-after})")

    # Winsorize (optional)
    if knobs["winsorize"]["enabled"]:
        for col in df.select_dtypes(include=[np.number]).columns:
            lower = df[col].quantile(knobs["winsorize"]["lower_q"])
            upper = df[col].quantile(knobs["winsorize"]["upper_q"])
            df[col] = np.clip(df[col], lower, upper)
        plan["actions"].append("winsorize")

        # Drop id-like columns (almost unique values)
    id_like_cols = []
    for col in data_card["columns"]:
        if col["name"] == target:
            continue

        total_rows = data_card["rows"]
        null_fraction = col["missing_pct"]
        non_null_count = int(total_rows * (1 - null_fraction))

        if non_null_count > 0:
            uniqueness_ratio = col["n_unique"] / non_null_count
        else:
            uniqueness_ratio = 0.0
        

        if uniqueness_ratio >= knobs.get("id_like_threshold", 0.95):
            if col["name"] in df.columns:
                df = df.drop(columns=[col["name"]])
                id_like_cols.append({
                    "name": col["name"],
                    "uniqueness_ratio": round(uniqueness_ratio, 3)
                })

    if id_like_cols:
        plan["id_like_columns"] = id_like_cols
        plan["dropped_columns"].extend(c["name"] for c in id_like_cols)


    # Save outputs
    clean_path = run_dir / "data_clean.parquet"
    df.to_parquet(clean_path, index=False)

    plan_path = run_dir / "cleaning_plan.json"
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)

    return clean_path, plan_path
