#!/usr/bin/env python

import pandas as pd
import argparse
from pathlib import Path
import neat
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--stats", default="eval-stats.csv")
args = parser.parse_args()
fn = Path(args.stats)
df = pd.read_csv(fn, names=["level", "agent", "fitness"])
print(df)
agg = df.groupby("agent")["fitness"].mean()
print(agg)
winner = agg.idxmax()
cp, i = winner.rsplit("-", 1)
d, cp = cp.split("-", 1)
print(cp, i)
cp_fn = fn.parent / cp
agents = list(neat.Checkpointer.restore_checkpoint(cp_fn).population.values())
winner_fn = fn.parent / "{}-{}.pickle".format(fn.parent.name, winner)
with open(winner_fn, "wb") as f:
    pickle.dump(agents[int(i)], f)
