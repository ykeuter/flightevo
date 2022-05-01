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
    def __init__(self, resolution_width, resolution_height, speed_x, speed_yz,
                 bounds):
        self._resolution_width = resolution_width
        self._resolution_height = resolution_height
        self._mlp = None
        self._device = "cuda"
        self._coords = self._get_coords()
        self._img_pub = rospy.Publisher(
            "/kingfisher/dodger/depth", Image, queue_size=1)
        self._cv_bridge = CvBridge()
        self._speed_x = speed_x
        self._speed_yz = speed_yz
        self._bounds = bounds  # min_y, max_y, min_z, max_z

    def load(self, cppn, cfg):
        del self._mlp
        self._mlp = Mlp2D.from_cppn(cppn, cfg, self._coords, self._device)

    def compute_command_vision_based(self, state, img):
        s = self._transform_state(state)
        i = self._transform_img(img)
        a = self._mlp.activate(torch.cat((s, i),))
        v = self._transform_activations(a)
        return AgileCommand(
            t=state.t, mode=2, yawrate=0, velocity=v)

    def _transform_activations(self, a):
        index = a.argmax().item()
        if index == 0:  # up
            vz = self._speed_yz
            vy = 0
        # elif index == 1:  # upper right
        #     vz = self._speed_yz
        #     vy = -self._speed_yz
        elif index == 1:  # right
            vz = 0
            vy = -self._speed_yz
        # elif index == 3:  # lower right
        #     vz = -self._speed_yz
        #     vy = -self._speed_yz
        elif index == 2:  # down
            vz = -self._speed_yz
            vy = 0
        # elif index == 5:  # lower left
        #     vz = -self._speed_yz
        #     vy = self._speed_yz
        elif index == 3:  # left
            vz = 0
            vy = self._speed_yz
        # elif index == 7:  # upper left
        #     vz = self._speed_yz
        #     vy = self._speed_yz
        elif index == 4:  # center
            vz = 0
            vy = 0
        vx = self._speed_x
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
        r = 5

        # inputs = []
        # z = 0
        state = [
            (0, r * 2, ),  # up
            (r * 2, 0, ),  # right
            (0, -r * 2, ),  # down
            (-r * 2, 0, ),  # left
        ]
        img = self._get_grid(
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
            # (r, r, ),  # upper right
            (r, 0, ),  # right
            # (r, -r, ),  # lower right
            (0, -r, ),  # down
            # (-r, -r, ),  # lower left
            (-r, 0, ),  # left
            # (-r, r, ),  # upper left
            (0, 0, ),  # center
        ]

        # return [inputs, hidden1, hidden2, outputs]
        return [state + img, outputs]

    @staticmethod
    def _get_grid(ncols, nrows, width, height):
        return [
            (
                (c / ncols + 1 / ncols / 2 - .5) * width,
                (r / nrows + 1 / nrows / 2 - .5) * -height,
                # c * width / ncols - width / 2,
                # -r * height / nrows + height / 2,
            )
            for r in range(nrows)
            for c in range(ncols)
        ]

    def _transform_img(self, img):
        r, c = img.shape
        k0 = int(r / self._resolution_height)
        k1 = int(c / self._resolution_width)
        # copy needed due to non-writeable nparray
        new_img = 1 - torch.tensor(img) \
            .unfold(0, k0, k0).unfold(1, k1, k1).amin((-1, -2),)

        # len_y = (self._bounds[1] - self._bounds[0]) / 2
        # mid_y = (self._bounds[1] + self._bounds[0]) / 2
        # len_z = (self._bounds[3] - self._bounds[2]) / 2
        # mid_z = (self._bounds[3] + self._bounds[2]) / 2
        # if state.pos[1] > mid_y:  # left
        #     new_img[:, :int(self._resolution_width / 2)].clamp_(
        #         (state.pos[1] - mid_y) / len_y)
        # else:  # right
        #     new_img[:, int(self._resolution_width / 2):].clamp_(
        #         (mid_y - state.pos[1]) / len_y)
        # if state.pos[2] > mid_z:  # up
        #     new_img[:int(self._resolution_height / 2), :].clamp_(
        #         (state.pos[2] - mid_z) / len_z)
        # else:  # down
        #     new_img[int(self._resolution_height / 2):, :].clamp_(
        #         (mid_z - state.pos[2]) / len_z)

        msg = self._cv_bridge.cv2_to_imgmsg(new_img.numpy())
        self._img_pub.publish(msg)
        # print(r, c)
        # cv2.imshow("depth resized", new_img.numpy())
        # cv2.waitKey()
        # new_img /= (self._resolution_width * self._resolution_height)
        return new_img.view(-1)
