import neat
import numpy as np
import os
import argparse
import random
import string
from ruamel.yaml import YAML, RoundTripDumper, dump
from flightgym import VisionEnv_v1

from flightevo import utils
from flightevo.mlp import Mlp
from neat.csv_reporter import CsvReporter
from neat.winner_reporter import WinnerReporter
from pathlib import Path


class VisionTrainer:
    def __init__(self, env_cfg, neat_cfg, log_dir):
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            neat_cfg,
        )
        self._population = neat.Population(config)
        self._population.add_reporter(neat.Checkpointer(
            1, None, str(Path(log_dir) / "checkpoint-")
        ))
        self._population.add_reporter(neat.StdOutReporter(True))
        self._population.add_reporter(CsvReporter(Path(log_dir)))
        self._population.add_reporter(WinnerReporter(Path(log_dir)))
        self._generator = iter(self._population)
        self._current_agent = None
        self._mlp = None

        with open(env_cfg) as f:
            config = YAML().load(f)
        self._env = VisionEnv_v1(dump(config, Dumper=RoundTripDumper), False)
        self._img_width = self._env.getImgWidth()
        self._img_height = self._env.getImgHeight()

        self._device = "cuda"
        self._coords = self._get_coords()
        self._frame_id = 0

    def run(self):
        state = np.zeros([1, 25], dtype=np.float64)
        img = np.zeros(
            [1, self._img_width * self._img_height], dtype=np.float32)
        obs = np.zeros([1, self._env.getObsDim()], dtype=np.float64)
        rew = np.zeros([1, self._env.getRewDim()], dtype=np.float64)
        done = np.zeros(1, dtype=bool)
        info = np.zeros(
            [1, len(self._env.getExtraInfoNames())], dtype=np.float64)

        os.system(os.environ["FLIGHTMARE_PATH"] +
                  "/flightrender/RPG_Flightmare.x86_64 &")
        self._env.connectUnity()
        try:
            self._reset()
            self._env.reset(obs)
            while True:
                self._env.updateUnity(self._frame_id)
                self._env.getDepthImage(img)
                self._env.getQuadState(state)
                self._current_agent.fitness = max(
                    self._current_agent.fitness, state[0, 1])
                actions = self._mlp.activate(img.reshape(-1)) \
                    .astype(np.float64).reshape(1, 4)
                self._env.step(actions, obs, rew, done, info)
                self._frame_id += 1
                if done[0]:
                    self._reset()
        except Exception as e:
            print(e)
            self._env.disconnectUnity()

    def _reset(self):
        self._current_agent = next(self._generator)
        self._current_agent.fitness = 0
        self._mlp = Mlp.from_cppn(self._current_agent, self._population.config,
                                  self._coords, self._device)
        self._frame_id = 0

    def _get_coords(self):
        grid = self._get_grid(self._img_width, self._img_height, 10, 10)
        inputs = [(x, y, 0) for x, y in grid]

        grid = self._get_grid(8, 8, 10, 10)
        layer = [(x, y, 1) for x, y in grid]

        # outputs (not sure about motor config)
        outputs = [(5, 5, 2), (-5, 5, 2), (-5, -5, 2), (5, -5, 2), ]

        return [inputs, layer, outputs]

    def _get_grid(self, ncols, nrows, width, height):
        return (
            (
                c * width / (ncols - 1) - width / 2,
                r * height / (nrows - 1) - height / 2
            )
            for c in range(ncols)
            for r in range(nrows)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--neat", default="cfg/neat.cfg")
    parser.add_argument("--env", default="cfg/env.yaml")
    parser.add_argument("--log", default="logs/" + "".join(
        random.choice(string.ascii_lowercase + string.digits)
        for _ in range(8)
    ))
    args = parser.parse_args()
    Path(args.log).mkdir()
    t = VisionTrainer(args.env, args.neat, args.log)
    t.run()
