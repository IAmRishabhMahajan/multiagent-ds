import os
import yaml
from src.utils import ensure_dir
from src.io import load_csv, summarize_df
import argparse
from pathlib import Path

def load_config():
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("Missing config/config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, help="Path to input CSV")
    args = parser.parse_args()

    cfg = load_config()
    artifacts_dir = Path(cfg.get("artifacts_dir", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        df = load_csv(args.csv)
        print(summarize_df(df))
    else:
        print("Setup OK")
        print(f"Artifacts dir: {artifacts_dir.resolve()}")

if __name__ == "__main__":
    main()
