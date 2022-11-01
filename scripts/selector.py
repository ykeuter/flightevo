#!/usr/bin/env python

import pandas as pd
import argparse
from pathlib import Path
import neat
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--stats", default="logs/bigrun/run-0/selection-stats.csv")
args = parser.parse_args()
fn = Path(args.stats)
df = pd.read_csv(fn, names=["environment", "agent", "time"])
print(df)
agg = df.groupby("agent")["time"].mean()
print(agg)
winner = agg.idxmax()
cp, i = winner.rsplit("-", 1)
print(cp, i)
cp_fn = fn.parent / cp
agents = list(neat.Checkpointer.restore_checkpoint(cp_fn).population.values())
winner_fn = fn.parent / "{}-{}.pickle".format(fn.parent.name, winner)
with open(winner_fn, "wb") as f:
    pickle.dump(agents[int(i)], f)
