import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

 # --- Numeric pipeline ---
def log_if_skewed(X: np.ndarray, log_skew_threshold: float, feature_plan: dict):
        X_out = X.copy().astype(float)  # make sure it's float so np.log1p works
        n_features = X_out.shape[1]

        for i in range(n_features):
            col_data = X_out[:, i]
            # skip if any negative values
            if np.nanmin(col_data) < 0:
                feature_plan["numeric"]["log_applied_skipped"].append(f"col_{i}")
                continue
            skew = pd.Series(col_data).skew(skipna=True)
            if abs(skew) > log_skew_threshold:
                X_out[:, i] = np.log1p(col_data)
                feature_plan["numeric"]["log_applied"].append(f"col_{i}")
        return X_out

# --- Datetime pipeline ---
def extract_parts(X: pd.DataFrame, knobs: dict, feature_plan: dict):
        new_feats = {}
        for col in X.columns:
            try:
                series = pd.to_datetime(X[col], errors="coerce")
                if "year" in knobs["features"]["datetime_parts"]:
                    new_feats[f"{col}_year"] = series.dt.year
                if "month" in knobs["features"]["datetime_parts"]:
                    new_feats[f"{col}_month"] = series.dt.month
                if "day" in knobs["features"]["datetime_parts"]:
                    new_feats[f"{col}_day"] = series.dt.day
                if "weekday" in knobs["features"]["datetime_parts"]:
                    new_feats[f"{col}_weekday"] = series.dt.weekday
                if "hour" in knobs["features"]["datetime_parts"]:
                    new_feats[f"{col}_hour"] = series.dt.hour
            except Exception:
                feature_plan["datetime"]["warnings"].append(f"Failed to parse {col}, fallback categorical")
        return pd.DataFrame(new_feats)

def build_feature_pipeline(run_dir: Path, knobs: dict, seed: int = 42):

    """
    Step 6: Feature Engineering
    Inputs: data_clean.parquet, spec.json, data_card.json
    Outputs: feature_plan.json, feature_pipeline.pkl, feature_sample.json
    """

    # --- Load inputs ---
    df = pd.read_parquet(run_dir / "data_clean.parquet")
    with open(run_dir / "spec.json") as f:
        spec = json.load(f)
    with open(run_dir / "data_card.json") as f:
        data_card = json.load(f)

    target = spec["target"]

    # --- Column grouping ---
    num_cols = df.select_dtypes(include=[np.number]).columns.drop([target, "_row_id"], errors="ignore").tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    dt_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    ignored = [c for c in ["_row_id", target] if c in df.columns]
    feature_plan = {
        "schema_version": "1.0",
        "groups": {"numeric": num_cols, "categorical": cat_cols, "datetime": dt_cols},
        "numeric": {
            "impute": knobs["cleaning"]["impute_numeric"],
            "log_skew_threshold": knobs["features"]["log_skew_threshold"],
            "log_applied": [],
            "log_applied_skipped": [],
            "scaler": "standard"
        },
        "categorical": {
            "impute": knobs["cleaning"]["impute_categorical"],
            "ohe": {
                "min_frequency_requested": knobs["features"]["one_hot_min_frequency"],
                "min_frequency_effective": None,
                "handle_unknown": "infrequent_if_exist",
                "max_features_cap": knobs["features"]["max_one_hot_features"]
            }
        },
        "datetime": {
            "parts": knobs["features"]["datetime_parts"],
            "drop_original": True,
            "warnings": []
        },
        "ignored": ignored,
        "notes": []
    }

   


    # --- Numeric pipeline ---
    num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy=knobs["cleaning"]["impute_numeric"])),
    ("log", FunctionTransformer(
        log_if_skewed,
        validate=False,
        kw_args={
            "log_skew_threshold": knobs["features"]["log_skew_threshold"],
            "feature_plan": feature_plan
        },
        feature_names_out="one-to-one"   
    )),
    ("scaler", StandardScaler())
    ])


    # --- Categorical pipeline ---
    # initial encoder with requested min_frequency
    ohe = OneHotEncoder(
        min_frequency=knobs["features"]["one_hot_min_frequency"],
        handle_unknown="infrequent_if_exist",
        sparse_output=False
    )
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent" if knobs["cleaning"]["impute_categorical"] == "most_frequent" else "constant",
                                  fill_value="missing")),
        ("onehot", ohe)
    ])

    

    dt_pipeline = Pipeline(steps=[
        ("extract", FunctionTransformer(extract_parts, validate=False, kw_args={"knobs": knobs, "feature_plan": feature_plan})),
    ])

    # --- Combine transformers ---
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipeline, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipeline, cat_cols))
    if dt_cols:
        transformers.append(("dt", dt_pipeline, dt_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("var_thresh", VarianceThreshold(0.0) if knobs["features"]["drop_zero_variance"] else "passthrough")
    ])

    # --- Fit small sample for preview ---
    sample = df.sample(min(len(df), 5000), random_state=seed)
    X_sample = sample.drop(columns=ignored, errors="ignore")

    pipeline.fit(X_sample)

    # --- Check OHE feature explosion ---
    if "cat" in preprocessor.named_transformers_:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        n_features = len(ohe.get_feature_names_out(cat_cols))
        cap = knobs["features"]["max_one_hot_features"]

        # adjust min_frequency upward until under cap
        effective_min_freq = knobs["features"]["one_hot_min_frequency"]
        while n_features > cap and effective_min_freq < 0.5:  # avoid infinite loop
            effective_min_freq *= 2
            ohe.set_params(min_frequency=effective_min_freq)
            ohe.fit(sample[cat_cols])
            n_features = len(ohe.get_feature_names_out(cat_cols))

        feature_plan["categorical"]["ohe"]["min_frequency_effective"] = effective_min_freq
        if n_features > cap:
            feature_plan["notes"].append(f"OHE features still {n_features} > cap {cap}, consider higher frequency")
            raise RuntimeError("OHE explosion: exceeded max_one_hot_features")
        else:
            feature_plan["categorical"]["ohe"]["min_frequency_effective"] = effective_min_freq

    # --- Save pipeline ---
    joblib.dump(pipeline, run_dir / "feature_pipeline.pkl")

    # --- Save feature plan ---
    with open(run_dir / "feature_plan.json", "w") as f:
        json.dump(feature_plan, f, indent=2)

    # --- Save feature sample preview ---
    X_trans = pipeline.transform(X_sample)
    if hasattr(pipeline.named_steps["preprocessor"], "get_feature_names_out"):
        feat_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    else:
        feat_names = [f"f{i}" for i in range(X_trans.shape[1])]

    zero_var_dropped = []
    if knobs["features"]["drop_zero_variance"]:
        # detect dropped features
        mask = pipeline.named_steps["var_thresh"].get_support()
        zero_var_dropped = [name for name, keep in zip(feat_names, mask) if not keep]
        feat_names = [name for name, keep in zip(feat_names, mask) if keep]

    feature_sample = {
        "schema_version": "1.0",
        "output_n_features": int(X_trans.shape[1] - len(zero_var_dropped)),
        "example_feature_names": feat_names[:20],
        "zero_variance_dropped": zero_var_dropped
    }
    with open(run_dir / "feature_sample.json", "w") as f:
        json.dump(feature_sample, f, indent=2)

    return run_dir / "feature_pipeline.pkl"
