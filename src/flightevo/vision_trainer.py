import neat
import numpy as np
import os
import argparse
import random
import string
import pickle
import shutil
# import torch
from ruamel.yaml import YAML, RoundTripDumper, dump
from flightgym import VisionEnv_v1
import torch

from flightevo.mlp import Mlp
from neat.csv_reporter import CsvReporter
from neat.winner_reporter import WinnerReporter
from pathlib import Path
from torchvision.transforms.functional import resize
from flightevo.utils import replace_config, reset_stagnation


class VisionTrainer:
    def __init__(self, env_cfg, neat_cfg, log_dir, winner_pickle, checkpoint):
        if winner_pickle:
            with open(winner_pickle, "rb") as f:
                w = pickle.load(f)
            self._generator = self._yield(w)
        else:
            Path(log_dir).mkdir()
            shutil.copy2(env_cfg, log_dir)
            shutil.copy2(neat_cfg, log_dir)
            self._neat_config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                neat_cfg,
            )
            if checkpoint:
                pop = neat.Checkpointer.restore_checkpoint(checkpoint)
                pop = replace_config(pop, self._neat_config)
                reset_stagnation(pop)
            else:
                pop = neat.Population(self._neat_config)

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
        self._resolution_width = config["inputs"]["resolution_width"]
        self._resolution_height = config["inputs"]["resolution_height"]
        self._sim_dt = config["simulation"]["sim_dt"]

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
                # self._current_agent.fitness = max(
                #     self._current_agent.fitness, obs[0, 0])
                self._current_agent.fitness = self._frame_id * self._sim_dt
                actions = self._mlp.activate(np.concatenate([
                    self._transform_obs(obs),
                    self._transform_img(img),
                ])).astype(np.float64).reshape(1, 4)
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

    def _transform_img(self, img):
        if self._resolution_height == 0 or self._resolution_width == 0:
            return np.array([])
        scaled_img = resize(
            torch.tensor(img.reshape(1, self._img_height, self._img_width),
                         device='cpu'),
            (self._resolution_height, self._resolution_width))
        return scaled_img.numpy().reshape(-1)

    def _transform_obs(self, obs):
        # obs: pos, eulerzyx, vel, omega
        v = obs.reshape(-1)
        return np.array([
            # position
            max(-v[1], .0),  # -y
            max(v[1], .0),  # y
            max(v[2], .0),  # z
            max(-v[2], .0),  # -z
            # velocity
            max(v[3], .0),  # x
            max(-v[3], .0),  # -x
            max(-v[4], .0),  # -y
            max(v[4], .0),  # y
            v[5],  # z
            # rot_x
            max(v[6], .0),  # x
            max(-v[6], .0),  # -x
            max(-v[7], .0),  # -y
            max(v[7], .0),  # y
            v[8],  # z
            # rot_y
            max(v[9], .0),  # x
            max(-v[9], .0),  # -x
            max(v[10], .0),  # y
            max(-v[10], .0),  # -y
            v[11],  # z
            # rot_z
            v[12],  # x
            max(-v[13], .0),  # -y
            max(v[13], .0),  # y
            max(v[14], .0),  # z
            max(-v[14], .0),  # -z
            # angular velocity
            max(v[15], .0),  # x
            max(-v[15], .0),  # -x
            max(v[16], .0),  # y
            max(-v[16], .0),  # -y
            max(v[17], .0),  # z
            max(v[17], .0),  # z
            max(-v[17], .0),  # -z
            max(-v[17], .0),  # -z
        ], dtype=np.float32)

    def _get_coords(self):
        r = 5

        inputs = []
        z = 0
        pos = [
            (r, 0, z),  # -y
            (-r, 0, z),  # y
            (0, r, z),  # z
            (0, -r, z),  # -z
        ]
        inputs += pos
        z = -1
        vel = [
            (0, r, z),  # x
            (0, -r, z),  # -x
            (r, 0, z),  # -y
            (-r, 0, z),  # y
            (0, 0, z),  # z
        ]
        inputs += vel
        z = -2
        rot_x = [
            (0, r, z),  # x
            (0, -r, z),  # -x
            (r, 0, z),  # -y
            (-r, 0, z),  # y
            (0, 0, z),  # z
        ]
        inputs += rot_x
        z = -3
        rot_y = [
            (0, r, z),  # x
            (0, -r, z),  # -x
            (r, 0, z),  # y
            (-r, 0, z),  # -y
            (0, 0, z),  # z
        ]
        inputs += rot_y
        z = -4
        rot_z = [
            (0, 0, z),  # x
            (r, 0, z),  # -y
            (-r, 0, z),  # y
            (0, r, z),  # z
            (0, -r, z),  # -z
        ]
        inputs += rot_z
        z = -5
        omega = [
            (r, 0, z),  # x
            (-r, 0, z),  # -x
            (0, r, z),  # y
            (0, -r, z),  # -y
            (r, r, z),  # z
            (-r, -r, z),  # z
            (r, -r, z),  # -z
            (-r, r, z),  # -z
        ]
        inputs += omega
        z = -6
        grid = self._get_grid(
            self._resolution_width, self._resolution_height, r * 2, r * 2)
        img = [(x, y, z) for x, y in grid]
        inputs += img

        hidden1 = []
        z = 1
        grid = self._get_grid(12, 12, r * 2, r * 2)
        layer1 = [(x, y, z) for x, y in grid]
        hidden1 += layer1
        z = 2
        layer2 = [(x, y, z) for x, y in grid]
        hidden1 += layer2

        hidden2 = []
        z = 3
        grid = self._get_grid(8, 8, r * 2, r * 2)
        layer1 = [(x, y, z) for x, y in grid]
        hidden2 += layer1
        z = 7
        layer2 = [(x, y, z) for x, y in grid]
        hidden2 += layer2

        z = 4
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
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--neat", default="cfg/neat.cfg")
    parser.add_argument("--env", default="cfg/env.yaml")
    parser.add_argument("--log", default="logs/" + "".join(
        random.choice(string.ascii_lowercase + string.digits)
        for _ in range(8)
    ))
    args = parser.parse_args()
    t = VisionTrainer(args.env, args.neat, args.log,
                      args.winner, args.checkpoint)
    t.run()
