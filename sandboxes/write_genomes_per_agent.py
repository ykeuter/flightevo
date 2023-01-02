import pickle
from pathlib import Path


DIR = Path("logs/bigrun-7")

for p in DIR.rglob("run-*.pickle"):
    print(p)
    with open(p, "rb") as fi:
        g = pickle.load(fi)
    nodes = list(g.nodes.values())
    nodes.sort(key=lambda x: -abs(x.weight))
    tot = sum(abs(n.weight) for n in nodes)
    for n in nodes:
        with open(DIR / "genomes.csv", "a") as fo:
            fo.write("{},{},{},{},{}\n".format(
                p.stem,
                abs(n.weight) / tot,
                n.weight,
                n.zoom,
                n.scale,
            ))
