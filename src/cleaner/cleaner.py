import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import spearmanr

def detect_id_like(col: pd.Series, knobs: dict) -> bool:
    """Check if column is a clean row ID (integer-like, monotonic/permutation)."""
    if not pd.api.types.is_integer_dtype(col.dropna()):
        return False

    non_null = col.dropna()
    unique_ratio = non_null.nunique() / len(non_null) if len(non_null) else 0

    if unique_ratio < knobs["id_unique_ratio"]:
        return False

    # Spearman correlation with index
    rho, _ = spearmanr(non_null, np.arange(len(non_null)))
    if rho >= knobs["id_spearman_with_index"]:
        return True

    # Check contiguous sequence (allow small tolerance)
    sorted_vals = np.sort(non_null.unique())
    diffs = np.diff(sorted_vals)
    contiguous_ratio = (diffs == 1).mean()
    if contiguous_ratio >= (1 - knobs["id_contiguity_tolerance"]):
        return True

    return False


def analyze_text_column(col: pd.Series) -> dict:
    """Lightweight analysis for quarantine manifest."""
    non_null = col.dropna().astype(str)
    if non_null.empty:
        return {}

    avg_len = non_null.str.len().mean()
    sample = "".join(non_null.sample(min(500, len(non_null)), random_state=42))

    alpha = sum(c.isalpha() for c in sample) / len(sample)
    digit = sum(c.isdigit() for c in sample) / len(sample)
    sep = sum(not c.isalnum() for c in sample) / len(sample)

    prefixes = non_null.str.extract(r"^([A-Za-z]+)").dropna()[0]
    top_prefixes = prefixes.value_counts(normalize=True).head(5).round(2).to_dict()

    return {
        "avg_len": round(avg_len, 2),
        "char_classes_pct": {"alpha": round(alpha, 2), "digit": round(digit, 2), "sep": round(sep, 2)},
        "top_prefixes": [{"token": k, "pct": float(v)} for k, v in top_prefixes.items()]
    }


def clean_dataframe(df: pd.DataFrame, target: str, run_dir: Path, knobs: dict):
    
    run_dir.mkdir(parents=True, exist_ok=True)
    
    plan = {"schema_version": "1.0", "actions": [], "warnings": []}
    quarantine_manifest = {"schema_version": "1.0", "columns": []}
    quarantine_cols = {}

    # --- Add row id
    df["_row_id"] = np.arange(len(df))

    # --- Drop high-missing columns
    for col in df.columns:
        if col == target or col == "_row_id":
            continue
        pct_missing = df[col].isna().mean()
        if pct_missing >= knobs["drop_missing_threshold"]:
            plan["actions"].append({
                "action": "drop_missing_col",
                "column": col,
                "reason": f"pct_missing={pct_missing:.2f} ≥ {knobs['drop_missing_threshold']}"
            })
            df = df.drop(columns=[col])
    
    # --- Normalize strings
    if knobs.get("trim_strings", True):
        for col in df.select_dtypes(include=["object", "string"]).columns:
            df[col] = df[col].astype(str).str.strip()
        plan["actions"].append({"action": "trim_strings"})

    if knobs.get("normalize_booleans", True):
        for col in df.select_dtypes(include=["object", "bool"]).columns:
            df[col] = df[col].astype(str).str.lower().map(
                {"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0}
            )
        plan["actions"].append({"action": "normalize_booleans"})
    
    # --- Identify ID-like and quarantine columns
    for col in df.columns:
        if col in [target, "_row_id"]:
            continue

        non_null = df[col].dropna()
        non_null_count = len(non_null)
        unique_ratio = non_null.nunique() / non_null_count if non_null_count else 0

        # Drop pure row ID
        if detect_id_like(non_null, knobs):
            plan["actions"].append({
                "action": "drop_id_column",
                "column": col,
                "reason": f"integer-like, unique_ratio={unique_ratio:.3f}, monotonic/permutation"
            })
            df = df.drop(columns=[col])
            continue

        # Quarantine near-unique / semi-structured
        if unique_ratio >= knobs["id_unique_ratio"] or df[col].dtype == object:
            qmeta = {
                "name": col,
                "reasons": ["near_unique" if unique_ratio >= knobs["id_unique_ratio"] else "semi_structured"],
                "unique_ratio": round(unique_ratio, 3)
            }
            qmeta.update(analyze_text_column(df[col]))
            quarantine_manifest["columns"].append(qmeta)
            quarantine_cols[col] = df[col]
            df = df.drop(columns=[col])
            plan["actions"].append({
                "action": "drop_to_quarantine",
                "column": col,
                "reason": f"near_unique/semi_structured (unique_ratio={unique_ratio:.3f})"
            })
    
    # --- Leakage guard: if col == target for ≥99% non-nulls
    for col in df.columns:
        if col in [target, "_row_id"]:
            continue
        non_null = df[[col, target]].dropna()
        if not non_null.empty:
            match_ratio = (non_null[col] == non_null[target]).mean()
            threshold = knobs.get("leakage_guard_match_threshold", 0.99)
            if match_ratio >= threshold:
                quarantine_manifest["columns"].append({
                    "name": col,
                    "reasons": ["leakage_guard"],
                    "unique_ratio": float(non_null[col].nunique() / len(non_null))
                })
                quarantine_cols[col] = df[col]
                df = df.drop(columns=[col])
                plan["actions"].append({
                    "action": "drop_to_quarantine",
                    "column": col,
                    "reason": f"leakage_guard match_ratio={match_ratio:.2f} ≥ {threshold}"
                })
    
    # --- Drop duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    n_dropped = before - len(df)
    if n_dropped:
        plan["actions"].append({"action": "drop_duplicate_rows", "params": {"n_dropped": n_dropped}})
    
    # --- Drop duplicate columns
    dupes = []
    for i, col1 in enumerate(df.columns):
        for col2 in df.columns[i + 1:]:
            if df[col1].equals(df[col2]):
                dupes.append(col2)
    if dupes:
        df = df.drop(columns=dupes)
        plan["actions"].append({"action": "drop_duplicate_columns", "params": {"columns": dupes}})
    
    # --- Imputation
    for col in df.columns:
        if col in [target, "_row_id"]:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            strategy = knobs["impute_numeric"]
            fill_val = df[col].median() if strategy == "median" else df[col].mean()
            n_imp = df[col].isna().sum()
            df[col] = df[col].fillna(fill_val)
            plan["actions"].append({
                "action": "impute_numeric",
                "column": col,
                "params": {"strategy": strategy, "n_imputed": int(n_imp)}
            })
        else:
            strategy = knobs["impute_categorical"]
            fill_val = (
                df[col].mode(dropna=True)[0] if strategy == "most_frequent" and not df[col].mode().empty else "missing"
            )
            n_imp = df[col].isna().sum()
            df[col] = df[col].fillna(fill_val)
            plan["actions"].append({
                "action": "impute_categorical",
                "column": col,
                "params": {"strategy": strategy, "n_imputed": int(n_imp)}
            })

    # --- Winsorize (optional)
    if knobs["winsorize"]["enabled"]:
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in [target, "_row_id"]:
                continue
            lower = df[col].quantile(knobs["winsorize"]["lower_q"])
            upper = df[col].quantile(knobs["winsorize"]["upper_q"])
            df[col] = np.clip(df[col], lower, upper)
        plan["actions"].append({"action": "winsorize"})

    # --- Save outputs
    clean_path = run_dir / "data_clean.parquet"
    df.to_parquet(clean_path, index=False)

    plan_path = run_dir / "cleaning_plan.json"
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)
    if quarantine_cols:
        quarantine_df = pd.DataFrame(quarantine_cols)
        quarantine_df["_row_id"] = np.arange(len(quarantine_df))
        quarantine_path = run_dir / "quarantine.parquet"
        quarantine_df.to_parquet(quarantine_path, index=False)

        quarantine_manifest_path = run_dir / "quarantine_manifest.json"
        with open(quarantine_manifest_path, "w") as f:
            json.dump(quarantine_manifest, f, indent=2)
    else:
        quarantine_path = quarantine_manifest_path = None
    
    return clean_path, plan_path, quarantine_path, quarantine_manifest_path
