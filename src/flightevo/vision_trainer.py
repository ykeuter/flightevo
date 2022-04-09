import neat
import numpy as np
import os
import argparse
import random
import string
import pickle
# import torch
from ruamel.yaml import YAML, RoundTripDumper, dump
from flightgym import VisionEnv_v1

from flightevo.mlp import Mlp
from neat.csv_reporter import CsvReporter
from neat.winner_reporter import WinnerReporter
from pathlib import Path


class VisionTrainer:
    def __init__(self, env_cfg, neat_cfg, log_dir, winner_pickle):
        self._neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            neat_cfg,
        )
        if winner_pickle:
            with open(winner_pickle, "rb") as f:
                w = pickle.load(f)
            self._generator = self._yield(w)
        else:
            pop = neat.Population(config)
            pop.add_reporter(neat.Checkpointer(
                1, None, str(Path(log_dir) / "checkpoint-")
            ))
            pop.add_reporter(neat.StdOutReporter(True))
            pop.add_reporter(CsvReporter(Path(log_dir)))
            pop.add_reporter(WinnerReporter(Path(log_dir)))
            self._generator = iter(pop)
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

    def _yield(self, x):
        while True:
            yield x

    def run(self):
        # state = np.zeros([1, 25], dtype=np.float64)
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
                # self._env.getQuadState(state)
                self._current_agent.fitness = max(
                    self._current_agent.fitness, obs[0, 0])
                actions = self._mlp.activate(
                    np.concatenate([img.reshape(-1), self._transform_obs(obs)])
                ).astype(np.float64).reshape(1, 4)
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
        del self._mlp
        self._mlp = Mlp.from_cppn(self._current_agent, self._neat_config,
                                  self._coords, self._device)
        # print(torch.cuda.memory_allocated())
        self._frame_id = 0

    def _transform_obs(self, obs):
        # obs: pos, eulerzyx, vel, omega
        v = obs.reshape(-1)
        return np.array([
            # position
            v[1],  # y
            -v[1],  # -y
            v[2],  # z
            # velocity
            v[6],  # x
            -v[6],  # -x
            v[7],  # y
            -v[7],  # -y
            v[8],  # z
            # euler angles
            v[5],  # x
            -v[5],  # -x
            v[4],  # y
            -v[4],  # -y
            v[3],  # z
            -v[3],  # -z
            # angular velocity
            v[9],  # x
            -v[9],  # -x
            v[10],  # y
            -v[10],  # -y
            v[11],  # z
            -v[11],  # -z
        ], dtype=np.float32)

    def _get_coords(self):
        r = 5

        z = 0
        grid = self._get_grid(self._img_width, self._img_height, r * 2, r * 2)
        img = [(x, y, z) for x, y in grid]
        pos = [
            (-r, 0, z),  # y
            (r, 0, z),  # -y
            (0, 0, z),  # z
        ]
        z = 1
        vel = [
            (0, r, z),  # x
            (0, -r, z),  # -x
            (-r, 0, z),  # y
            (r, 0, z),  # -y
            (0, 0, z),  # z
        ]
        z = 2
        rot = [
            (-r, r, z),  # x
            (r, r, z),  # -x
            (0, r, z),  # y
            (0, -r, z),  # -y
            (-r, -r, z),  # z
            (r, -r, z),  # -z
        ]
        z = 3
        omega = [
            (-r, r, z),  # x
            (r, r, z),  # -x
            (0, r, z),  # y
            (0, -r, z),  # -y
            (-r, -r, z),  # z
            (r, -r, z),  # -z
        ]
        inputs = img + pos + vel + rot + omega

        z = 4
        grid = self._get_grid(12, 12, r * 2, r * 2)
        layer1 = [(x, y, z) for x, y in grid]
        z = 5
        layer2 = [(x, y, z) for x, y in grid]
        hidden1 = layer1 + layer2

        z = 6
        grid = self._get_grid(8, 8, r * 2, r * 2)
        layer1 = [(x, y, z) for x, y in grid]
        z = 7
        layer2 = [(x, y, z) for x, y in grid]
        hidden2 = layer1 + layer2

        z = 8
        outputs = [
            (r, r, z),  # fr
            (-r, -r, z),  # bl
            (r, -r, z),  # br
            (-r, r, z),  # fl
        ]

        return [inputs, hidden1, hidden2, outputs]

    def _get_grid(self, ncols, nrows, width, height):
        return [
            (
                c * width / ncols - width / 2,
                -r * height / nrows + height / 2
            )
            for r in range(nrows)
            for c in range(ncols)
        ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--winner", default="")
    parser.add_argument("--neat", default="cfg/neat.cfg")
    parser.add_argument("--env", default="cfg/env.yaml")
    parser.add_argument("--log", default="logs/" + "".join(
        random.choice(string.ascii_lowercase + string.digits)
        for _ in range(8)
    ))
    args = parser.parse_args()
    Path(args.log).mkdir()
    t = VisionTrainer(args.env, args.neat, args.log, args.winner)
    t.run()
