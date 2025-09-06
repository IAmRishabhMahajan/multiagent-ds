import os
import yaml, json, argparse
from src.utils.common import ensure_dir
import argparse
from pathlib import Path
from src.io.io import load_csv, summarize_df
from src.spec.builder import build_spec
from src.profiler.profiler import profile_dataframe
from src.utils.common import load_config, load_knobs
from src.cleaner.cleaner import clean_dataframe




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--task", type=str, choices=["classification", "regression"])
    parser.add_argument("--metric", type=str)
    args = parser.parse_args()

    cfg = load_config()
    artifacts_dir = Path(cfg.get("artifacts_dir", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    knobs = load_knobs()

    #step 3 implementing a json file to store the spec
    
    df = load_csv(args.csv)
    print(summarize_df(df))
        
    spec_path = build_spec(df, args.target, cfg.get("seed", 42), artifacts_dir,
                                   override_task=args.task, override_metric=args.metric)
    print(f"Spec saved: {spec_path}")
    print(json.load(open(spec_path)))
    
    
    #step 4 data profiling using knobs
    run_dir = spec_path.parent
    data_card_path = run_dir / "data_card.json"
    profile_dataframe(df, args.target, data_card_path, knobs["profiling"])
    print(f"Data card saved: {data_card_path}")

    # Step 5: cleaner
    clean_path, plan_path = clean_dataframe(df, args.target, run_dir, knobs["cleaning"])
    print(f"Cleaned data saved: {clean_path}")
    print(f"Cleaning plan saved: {plan_path}")



if __name__ == "__main__":
    main()
