#!/bin/bash

DIR="logs/bigrun-3"

for i in {0..19}
do
  # training
  env="$DIR/env-training.yaml"
  neat="$DIR/neat.cfg"
  log="$DIR/run-$i"
  python -m flightevo.dodge_trainer --neat $neat --env $env --log $log
  # evaluation for selection
  out="$log/selection-stats.csv"
  env="$DIR/env-selection.yaml"
  checkpoint="$(ls $log/checkpoint-* | sort -Vr | head -1)"
  echo $checkpoint
  python -m flightevo.evaluator --out $out --env $env --checkpoint $checkpoint
  # selection
  ./scripts/selector.py --stats $out
  # evaluation
  out="$DIR/evaluation-stats.csv"
  env="$DIR/env-evaluation.yaml"
  agent="$(ls $log/run-$i-checkpoint-*.pickle)"
  python -m flightevo.evaluator --out $out --env $env --agent $agent
done