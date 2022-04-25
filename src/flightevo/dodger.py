from collections import namedtuple
import cv2
import torch

from flightevo.mlp2d import Mlp2D


AgileCommand = namedtuple("AgileCommand", ["mode", "velocity", "yawrate", "t"])
AgileQuadState = namedtuple("AgileCommand", ["t"])


class Dodger:
    MAX_SPEED = 1.0

    def __init__(self, resolution_width, resolution_height):
        self._resolution_width = resolution_width
        self._resolution_height = resolution_height
        self._mlp = None
        self._device = "cuda"
        self._coords = self._get_coords()

    def load(self, cppn, cfg):
        del self._mlp
        self._mlp = Mlp2D.from_cppn(cppn, cfg, self._coords, self._device)

    def compute_command_vision_based(self, state, img):
        i = self._transform_img(img)
        a = self._mlp.activate(i)  # up, down, right, left, forward
        vx = a[4] * self.MAX_SPEED
        vy = (a[3] if a[3] > a[2] else -a[2]) * self.MAX_SPEED
        vz = (a[0] if a[0] > a[1] else -a[1]) * self.MAX_SPEED
        return AgileCommand(
            t=state.t, mode=2, yawrate=0, velocity=[vx, vy, vz])

    def _get_coords(self):
        r = 5

        inputs = []
        # z = 0
        grid = self._get_grid(
            self._resolution_width, self._resolution_height, r * 2, r * 2)
        # img = [(x, y, z) for x, y in grid]
        # inputs += img

        # hidden1 = []
        # z = 1
        # grid = self._get_grid(4, 4, r * 2, r * 2)  # max 12x12
        # layer1 = [(x, y, z) for x, y in grid]
        # hidden1 += layer1
        # z = 2
        # layer2 = [(x, y, z) for x, y in grid]
        # hidden1 += layer2

        # hidden2 = []
        # z = 3
        # grid = self._get_grid(4, 4, r * 2, r * 2)  # max 8x8
        # layer1 = [(x, y, z) for x, y in grid]
        # hidden2 += layer1
        # z = 4
        # layer2 = [(x, y, z) for x, y in grid]
        # hidden2 += layer2

        # z = 5
        outputs = [
            (0, r, ),  # up
            (0, -r, ),  # down
            (r, 0, ),  # right
            (-r, 0, ),  # left
            (0, 0, ),  # forward
        ]

        # return [inputs, hidden1, hidden2, outputs]
        return [grid, outputs]

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
        r, c = img.shape
        k0 = int(r / self._resolution_height)
        k1 = int(c / self._resolution_width)
        # copy needed due to non-writeable nparray
        new_img = torch.tensor(img) \
            .unfold(0, k0, k0).unfold(1, k1, k1).amin((-1, -2),)
        # print(r, c)
        # cv2.imshow("depth resized", new_img.numpy())
        # cv2.waitKey()
        return new_img.view(-1)
