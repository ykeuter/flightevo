#!/bin/bash

DIR="logs/bigrun"

for i in {0..19}
do
  # training
  env="$DIR/env-training.yaml"
  neat="$DIR/neat.cfg"
  log="$DIR/run-$i"
  python -m flightevo.dodge_trainer --neat $neat --env $env --log $log
  # evaluatin for selection
  out="$log/selection-stats.csv"
  env="$DIR/env-selection.yaml"
  checkpoint="$(ls $log/environment_*.pickle | sort -Vr | head -1)"
  python -m flightevo.evaluator --out $out --env $env --checkpoint $checkpoint
done