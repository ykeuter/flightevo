from collections import namedtuple
import cv2
import random
import rospy
import torch
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge

from flightevo.mlp2d import Mlp2D
from flightevo.utils import AgileCommand


class Dodger:
    BORDER = 0

    def __init__(self, resolution_width, resolution_height,
                 speed_x, speed_y, speed_z, bounds, gamma, acc, margin):
        self._resolution_width = resolution_width
        self._resolution_height = resolution_height
        self._mlp = None
        self._device = "cpu"
        self._coords = self._get_coords()
        # self._img_pub = rospy.Publisher(
        #     "/kingfisher/dodger/depth", Image, queue_size=1)
        # self._cv_bridge = CvBridge()
        self._speed_x = speed_x
        self._speed_y = speed_y
        self._speed_z = speed_z
        self._gamma = gamma
        self._bounds = bounds  # min_y, max_y, min_z, max_z
        self._acc = acc
        self._margin = margin

    def load(self, cppn):
        del self._mlp
        self._mlp = Mlp2D.from_cppn(cppn, self._coords, self._device)

    def compute_command_vision_based(self, state, img):
        # s = self._transform_state(state)
        i = self._transform_img(img, state)
        a = self._mlp.activate(i)
        v = self._transform_activations(a, state)
        c = AgileCommand(2)
        c.t = state.t
        c.velocity = v
        return c

    def _transform_activations(self, a, state):
        # a: up, right, down, left, center
        if state.pos[1] < self._bounds[0] + self._margin:  # avoid right
            a[1] = -float("inf")
        if state.pos[1] > self._bounds[1] - self._margin:  # avoid left
            a[3] = -float("inf")
        if state.pos[2] < self._bounds[2] + self._margin:  # avoid down
            a[2] = -float("inf")
        if state.pos[2] > self._bounds[3] - self._margin:  # avoid up
            a[0] = -float("inf")

        vy, vz = 0, 0
        vx = min(self._speed_x, state.vel[0] + self._acc)
        index = a.argmax().item()
        if index == 0:  # up
            vz = self._speed_z
        elif index == 1:  # right
            vy = -self._speed_y
        elif index == 2:  # down
            vz = -self._speed_z
        elif index == 3:  # left
            vy = self._speed_y
        return [vx, vy, vz]

    def _transform_state(self, state):
        s = torch.zeros(4, dtype=torch.float32)  # up, right, down, left

        len_y = (self._bounds[1] - self._bounds[0]) / 2
        mid_y = (self._bounds[1] + self._bounds[0]) / 2
        len_z = (self._bounds[3] - self._bounds[2]) / 2
        mid_z = (self._bounds[3] + self._bounds[2]) / 2
        if state.pos[1] > mid_y:  # y
            s[3] = (state.pos[1] - mid_y) / len_y
        else:
            s[1] = (mid_y - state.pos[1]) / len_y
        if state.pos[2] > mid_z:  # z
            s[0] = (state.pos[2] - mid_z) / len_z
        else:
            s[2] = (mid_z - state.pos[2]) / len_z
        return s

    def _get_coords(self):
        r = 10

        img = self._get_grid(
            self._resolution_width + self.BORDER * 2,
            self._resolution_height + self.BORDER * 2,
            r * 2,
            r * 2
        )

        outputs = [
            (0, r, ),  # up
            (r, 0, ),  # right
            (0, -r, ),  # down
            (-r, 0, ),  # left
            (0, 0, ),  # center
        ]

        return [img, outputs]

    @staticmethod
    def _get_grid(ncols, nrows, width, height):
        return [
            (
                (c / ncols + 1 / ncols / 2 - .5) * width,
                (r / nrows + 1 / nrows / 2 - .5) * -height,
            )
            for r in range(nrows)
            for c in range(ncols)
        ]

    def _transform_img(self, img, state):
        r, c = img.shape
        k0 = int(r / self._resolution_height)
        k1 = int(c / self._resolution_width)
        # copy needed due to non-writeable nparray
        new_img = 1 - torch.tensor(img) \
            .unfold(0, k0, k0).unfold(1, k1, k1).amin((-1, -2),)

        # add border
        right = max(state.pos[1] - self._bounds[0] - 1, .0)
        left = max(self._bounds[1] - state.pos[1] - 1, .0)
        down = max(state.pos[2] - self._bounds[2] - 1, .0)
        up = max(self._bounds[3] - state.pos[2] - 1, .0)

        bw = int(self._resolution_width / 4)
        bh = int(self._resolution_height / 4)
        new_img[:bh, :].clamp_(1 - up / 100)
        new_img[-bh:, :].clamp_(1 - down / 100)
        new_img[:, :bw].clamp_(1 - left / 100)
        new_img[:, -bw:].clamp_(1 - right / 100)

        # non-linear scaling
        new_img.pow_(self._gamma)

        # msg = self._cv_bridge.cv2_to_imgmsg(new_img.numpy())
        # self._img_pub.publish(msg)

        return new_img.view(-1)
