# src/io.py
import pandas as pd
from pathlib import Path

def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV safely with pandas and basic checks."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV {path}: {e}")
    
    if df.empty:
        raise ValueError(f"CSV {path} is empty.")
    return df

def summarize_df(df: pd.DataFrame) -> str:
    """Return a short summary: rows, cols, head."""
    rows, cols = df.shape
    preview = df.head().to_string()
    return f"Rows: {rows}, Cols: {cols}\n\nHead:\n{preview}\n"
