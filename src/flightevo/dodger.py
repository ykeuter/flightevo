from collections import namedtuple
import cv2
import rospy
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

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
        self._img_pub = rospy.Publisher(
            "/kingfisher/dodger/depth", Image, queue_size=1)
        self._cv_bridge = CvBridge()

    def load(self, cppn, cfg):
        del self._mlp
        self._mlp = Mlp2D.from_cppn(cppn, cfg, self._coords, self._device)

    def compute_command_vision_based(self, state, img):
        i = self._transform_img(img)

        a = self._mlp.activate(i)

        index = a.argmax().item()
        if index == 0:  # up
            vz = self.MAX_SPEED
            vy = 0
        elif index == 1:  # upper right
            vz = self.MAX_SPEED
            vy = -self.MAX_SPEED
        elif index == 2:  # right
            vz = 0
            vy = -self.MAX_SPEED
        elif index == 3:  # lower right
            vz = -self.MAX_SPEED
            vy = -self.MAX_SPEED
        elif index == 4:  # down
            vz = -self.MAX_SPEED
            vy = 0
        elif index == 5:  # lower left
            vz = -self.MAX_SPEED
            vy = self.MAX_SPEED
        elif index == 6:  # left
            vz = 0
            vy = self.MAX_SPEED
        elif index == 7:  # upper left
            vz = self.MAX_SPEED
            vy = self.MAX_SPEED
        elif index == 8:  # center
            vz = 0
            vy = 0

        vx = self.MAX_SPEED
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
            (r, r, ),  # upper right
            (r, 0, ),  # right
            (r, -r, ),  # lower right
            (0, -r, ),  # down
            (-r, -r, ),  # lower left
            (-r, 0, ),  # left
            (-r, r, ),  # upper left
            (0, 0, ),  # center
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
        msg = self._cv_bridge.cv2_to_imgmsg(new_img.numpy())
        self._img_pub.publish(msg)
        # print(r, c)
        # cv2.imshow("depth resized", new_img.numpy())
        # cv2.waitKey()
        return new_img.view(-1)
