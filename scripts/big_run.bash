#!/bin/bash

DIR="logs/bigrun"
env="$DIR/env-training.yaml"
neat="$DIR/neat.cfg"
for i in {0..19}
do
  log="$DIR/run-$i"
  python -m fligtevo.dodge_trainer --neat $neat --env $env --log $log
done