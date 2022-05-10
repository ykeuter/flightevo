from collections import namedtuple
import cv2
import rospy
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from flightevo.mlp2d import Mlp2D


AgileCommand = namedtuple("AgileCommand", ["mode", "velocity", "yawrate", "t"])
AgileQuadState = namedtuple("AgileCommand", ["t", "pos"])


class Dodger:
    BORDER = 5

    def __init__(self, resolution_width, resolution_height,
                 speed_x, speed_y, speed_z, bounds, gamma):
        self._resolution_width = resolution_width
        self._resolution_height = resolution_height
        self._mlp = None
        self._device = "cuda"
        self._coords = self._get_coords()
        self._img_pub = rospy.Publisher(
            "/kingfisher/dodger/depth", Image, queue_size=1)
        self._cv_bridge = CvBridge()
        self._speed_x = speed_x
        self._speed_y = speed_y
        self._speed_z = speed_z
        self._gamma = gamma
        self._bounds = bounds  # min_y, max_y, min_z, max_z

    def load(self, cppn):
        del self._mlp
        self._mlp = Mlp2D.from_cppn(cppn, self._coords, self._device)

    def compute_command_vision_based(self, state, img):
        # s = self._transform_state(state)
        i = self._transform_img(img, state)
        a = self._mlp.activate(i)
        v = self._transform_activations(a)

        return AgileCommand(
            t=state.t, mode=2, yawrate=0, velocity=v)

    def _transform_activations(self, a):
        # a: up, right, down, left, center
        # if state.pos[1] < self._bounds[0] + 1:  # avoid right
        #     a[1] = -float("inf")
        # if state.pos[1] > self._bounds[1] - 1:  # avoid left
        #     a[3] = -float("inf")
        # if state.pos[2] < self._bounds[2] + 1:  # avoid down
        #     a[2] = -float("inf")
        # if state.pos[2] > self._bounds[3] - 1:  # avoid up
        #     a[0] = -float("inf")

        vy, vz = 0, 0
        vx = self._speed_x
        # if state.pos[0] < 3.:
        #     vx *= .5
        index = a.argmax().item()
        if index == 0:  # up
            vz = self._speed_z
        elif index == 1:  # right
            vy = -self._speed_y
        elif index == 2:  # down
            vz = -self._speed_z
        elif index == 3:  # left
            vy = self._speed_y
        # elif index == 4:  # center

        # if a[0] > a[2]:
        #     vz = a[0].item() * self._speed_z
        # else:
        #     vz = -a[2].item() * self._speed_z
        # if a[1] > a[3]:
        #     vy = -a[1].item() * self._speed_y
        # else:
        #     vy = a[3].item() * self._speed_y
        # vx = a[4].item() * self._speed_x

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

        # state = [
        #     (0, r * 2, ),  # up
        #     (r * 2, 0, ),  # right
        #     (0, -r * 2, ),  # down
        #     (-r * 2, 0, ),  # left
        # ]
        img = self._get_grid(
            self._resolution_width + self.BORDER * 2,
            self._resolution_height + self.BORDER * 2,
            r * 2,
            r * 2
        )

        outputs = [
            (0, r, ),  # up
            # (r, r, ),  # upper right
            (r, 0, ),  # right
            # (r, -r, ),  # lower right
            (0, -r, ),  # down
            # (-r, -r, ),  # lower left
            (-r, 0, ),  # left
            # (-r, r, ),  # upper left
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
        right = state.pos[1] - self._bounds[0]
        left = self._bounds[1] - state.pos[1]
        down = state.pos[2] - self._bounds[2]
        up = self._bounds[3] - state.pos[2]
        h, w, b = self._resolution_height, self._resolution_width, self.BORDER
        new_img = torch.hstack((
            torch.full((h + 2 * b, b), 1 - left / 100),
            torch.vstack((
                torch.full((b, w), 1 - up / 100),
                new_img,
                torch.full((b, w), 1 - down / 100),
            )),
            torch.full((h + 2 * b, b), 1 - right / 100),
        ))
        # non-linear scaling
        new_img.pow_(self._gamma)

        msg = self._cv_bridge.cv2_to_imgmsg(new_img.numpy())
        self._img_pub.publish(msg)

        return new_img.view(-1)
