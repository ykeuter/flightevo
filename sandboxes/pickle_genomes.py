import pickle
import torch
import numpy as np
import neat
from pathlib import Path

cp = Path("logs/eval_fast/checkpoint-4")
pop = neat.Checkpointer.restore_checkpoint(cp)
genomes = list(pop.population.values())

for i, g in enumerate(genomes):
    fn = cp.parent / "member-{}.pickle".format(i)
    with open(fn, "wb") as f:
        pickle.dump(g, f)
