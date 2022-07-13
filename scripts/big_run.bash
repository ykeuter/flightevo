#!/bin/bash

DIR=$1

for ((i=${2:-0}; i<=${3:-19}; i++))
do
  # training
  env="$DIR/env-training.yaml"
  neat="$DIR/neat.cfg"
  log="$DIR/run-$i"
  if [[ -f "$env"]]
  then
    python -m flightevo.dodge_trainer --neat $neat --env $env --log $log
  fi
  # evaluation for selection
  out="$log/selection-stats.csv"
  env="$DIR/env-selection.yaml"
  checkpoint="$(ls $log/checkpoint-* | sort -Vr | head -1)"
  if [[ -f "$env"]]
  then
    python -m flightevo.evaluator --out $out --env $env --checkpoint $checkpoint
    # selection
    ./scripts/selector.py --stats $out
  fi
  # evaluation
  out="$DIR/evaluation-stats.csv"
  env="$DIR/env-evaluation.yaml"
  agent="$(ls $log/run-$i-checkpoint-*.pickle)"
  if [[ -f "$env"]]
  then
    python -m flightevo.evaluator --out $out --env $env --agent $agent
  fi
done