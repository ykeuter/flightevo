#!/bin/bash

DIR=$1

for ((i=${2:-0}; i<=${3:-19}; i++))
do
  log="$DIR/run-$i"
  # training
  env="$DIR/env-training.yaml"
  if [[ -f "$env" ]]
  then
    neat="$DIR/neat.cfg"
    python -m flightevo.dodge_trainer --neat $neat --env $env --log $log --seed $i
  fi
  # evaluation for selection
  env="$DIR/env-selection.yaml"
  if [[ -f "$env" ]]
  then
    out="$log/selection-stats.csv"
    checkpoint="$(ls $log/checkpoint-* | sort -Vr | head -1)"
    python -m flightevo.evaluator --out $out --env $env --checkpoint $checkpoint
    # selection
    ./scripts/selector.py --stats $out
  fi
  # evaluation
  env="$DIR/env-evaluation.yaml"
  if [[ -f "$env" ]]
  then
    out="$DIR/evaluation-stats.csv"
    # agent="$(ls $log/run-$i-checkpoint-*.pickle)"
    checkpoint="$(ls $log/checkpoint-* | sort -Vr | head -1)"
    python -m flightevo.evaluator --out $out --env $env --checkpoint $checkpoint
  fi
done