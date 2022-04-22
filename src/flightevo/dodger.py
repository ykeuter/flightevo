from collections import namedtuple
from torchvision.transforms.functional import resize
import torch

from flightevo.mlp import Mlp


AgileCommand = namedtuple("AgileCommand", ["mode", "velocity", "yawrate", "t"])
AgileQuadState = namedtuple("AgileCommand", ["t"])


class Dodger:
    SPEED_X = 1.0
    MAX_SPEED = 3.0

    def __init__(self, resolution_width, resolution_height):
        self._resolution_width = resolution_width
        self._resolution_height = resolution_height
        self._mlp = None
        self._device = "cuda"
        self._coords = self._get_coords()

    def load(self, cppn, cfg):
        del self._mlp
        self._mlp = Mlp.from_cppn(cppn, cfg, self._coords, self._device)

    def compute_command_vision_based(self, state, img):
        a = self._mlp.activate(img)  # up, down, right, left
        vx = self.SPEED_X
        vy = (a[3] if a[3] > a[2] else -a[2]) * self.MAX_SPEED
        vz = (a[0] if a[0] > a[1] else -a[1]) * self.MAX_SPEED
        return AgileCommand(
            t=state.t, mode=2, yawrate=0, velocity=[vx, vy, vz])

    def _get_coords(self):
        r = 5

        inputs = []
        z = 0
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
        z = 4
        layer2 = [(x, y, z) for x, y in grid]
        hidden2 += layer2

        z = 5
        outputs = [
            (0, r, z),  # up
            (0, -r, z),  # down
            (r, 0, z),  # right
            (-r, 0, z),  # left
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

    def _transform_img(self, img):
        if self._resolution_height == 0 or self._resolution_width == 0:
            return np.array([])
        scaled_img = resize(
            torch.tensor(img.reshape(1, self._img_height, self._img_width),
                         device='cpu'),
            (self._resolution_height, self._resolution_width))
        return scaled_img.numpy().reshape(-1)
