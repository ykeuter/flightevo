import pickle
from pathlib import Path
import pandas as pd


DIR = Path("logs/bigrun-11")

dfs = []
for d in DIR.iterdir():
    if not d.is_dir():
        continue
    df = pd.read_csv(d / "stats.csv")
    df["run"] = d.stem
    dfs.append(df)
dfs = pd.concat(dfs)
dfs.to_csv(DIR / "training-stats.csv", index=False)
