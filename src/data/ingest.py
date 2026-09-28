from pathlib import Path
import pandas as pd

from src.data.config import RAW, RAW_FILE, TEXT, DATE, LABEL, KEEP


def load_raw(path: Path | None = None) -> pd.DataFrame:
    path = path or (RAW / RAW_FILE)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Put the export in data/raw/")

    cols = list(pd.read_csv(path, nrows=0).columns)
    usecols = [c for c in KEEP if c in cols]
    if missing := [c for c in KEEP if c not in cols]:
        print(f"[warn] not in file, skipping: {missing}")

    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df[DATE] = pd.to_datetime(df[DATE], errors="coerce")

    if (bad := df[DATE].isna().sum()):
        print(f"[warn] {bad} unparseable dates")

    df["year"] = df[DATE].dt.year
    df["month"] = df[DATE].dt.to_period("M")

    print(f"[load] {len(df):,} rows x {len(df.columns)} cols | "
          f"{df[DATE].min().date()} to {df[DATE].max().date()}")
    return df


if __name__ == "__main__":
    d = load_raw()
    print(d[LABEL].value_counts().to_string())