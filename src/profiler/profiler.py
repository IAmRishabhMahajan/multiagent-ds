# src/profiler/profiler.py
import pandas as pd
import numpy as np
import json
from pathlib import Path

def profile_dataframe(df: pd.DataFrame, target: str, out_path: Path, knobs: dict) -> Path:
    profile = {}

    # Basic shape
    profile["rows"], profile["cols"] = df.shape

    # Per-column stats
    columns = []
    for col in df.columns:
        col_data = df[col]
        info = {
            "name": col,
            "dtype": str(col_data.dtype),
            "missing_pct": float(col_data.isna().mean()),
            "n_unique": int(col_data.nunique(dropna=True))
        }
        if info["n_unique"] > knobs["high_cardinality_cat"]:
            info["warning"] = "High cardinality"
        columns.append(info)
    profile["columns"] = columns

    # Target summary
    if target in df.columns:
        tgt = df[target]
        if pd.api.types.is_numeric_dtype(tgt) and tgt.nunique() > 10:
            profile["target_summary"] = {
                "mean": float(tgt.mean()),
                "std": float(tgt.std()),
                "min": float(tgt.min()),
                "max": float(tgt.max()),
            }
        else:
            counts = tgt.value_counts(normalize=True).round(3).to_dict()
            profile["target_summary"] = {str(k): float(v) for k, v in counts.items()}
            minority = min(counts.values())
            if minority < knobs["imbalance_warn_threshold"]:
                profile.setdefault("warnings", []).append("Target imbalance detected")

    # Correlation matrix (numeric only, sampled)
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        sample_df = num_df.sample(min(knobs["corr_sample_rows"], len(num_df)), random_state=42)
        corr = sample_df.corr().round(3).to_dict()
        profile["correlations"] = corr

        # Strong correlation warning
        for i, col1 in enumerate(corr):
            for col2, val in corr[col1].items():
                if i < list(corr).index(col2) and abs(val) > knobs["strong_corr_threshold"]:
                    profile.setdefault("warnings", []).append(
                        f"Strong correlation {col1} ~ {col2} ({val})"
                    )

    # Save JSON
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2)

    return out_path
