from pathlib import Path
import neat


DIR = Path("logs/bigrun-16")

with open(DIR / "genomes.csv", "a") as fo:
    fo.write("run,generation,member,node,weight,zoom,scale,fitness\n")
for cp in DIR.rglob("checkpoint-*"):
    print(cp)
    pop = neat.Checkpointer.restore_checkpoint(cp)
    for i, g in enumerate(pop.population.values()):
        for j, n in enumerate(g.nodes.values()):
            with open(DIR / "genomes.csv", "a") as fo:
                fo.write(("{}," * 7 + "{}\n").format(
                    cp.parent.stem,
                    cp.stem,
                    i,
                    j,
                    n.weight,
                    n.zoom,
                    n.scale,
                    g.fitness,
                ))
