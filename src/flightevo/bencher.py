from flightevo.dodger import Dodger
from flightevo import utils
import numpy as np
import quaternion


class Bencher(Dodger):
    def __init__(self, resolution_width, resolution_height,
                 speed_x, speed_y, speed_z, bounds, gamma, acc, margin,
                 creep_z, creep_yaw):
        self._creep_z = creep_z
        self._creep_yaw = creep_yaw
        self._target = np.array([0., 0., 0.], dtype=np.float32)
        super().__init__(resolution_width, resolution_height,
                         speed_x, speed_y, speed_z, bounds, gamma, acc, margin)

    def set_target(self, t):
        # print("target: {}".format(t))
        self._target = t

    def compute_command_vision_based(self, state, img):
        # s = self._transform_state(state)
        i = self._transform_img(img, state)
        a = self._mlp.activate(i)
        v = self._transform_activations(a, state)
        v = self._adjust_z(v, state)
        yawrate = self._adjust_yaw(state)

        # TESTING
        # always go fwd
        a = np.array([0, 0, 0, 0, 1])
        v = self._transform_activations(a, state)
        # calc angle
        t = self._target - state.pos
        d = np.linalg.norm(t)
        rq = np.quaternion(*state.att)
        yq = np.quaternion(0, 0, 1, 0)
        y = rq * yq * np.conjugate(rq)
        a = np.arccos(np.dot(t, y.imag) / d)
        if a > .5 or d < 5:
            v = [x / 10 for x in v]
        v = self._adjust_z(v, state)
        yawrate = self._adjust_yaw(state)
        # print("pos: {}".format(state.pos))
        # print("v: {}".format(v))
        # TESTING

        c = utils.AgileCommand(2)
        c.t = state.t
        c.velocity = v
        c.yawrate = yawrate
        return c

    def _adjust_z(self, v, state):
        if state.pos[2] > self._target[2]:
            v[2] -= self._creep_z
        else:
            v[2] += self._creep_z
        return v

    def _adjust_yaw(self, state):
        t = self._target - state.pos
        rq = np.quaternion(*state.att)
        xq = np.quaternion(0, 1, 0, 0)
        x = rq * xq * np.conjugate(rq)
        a = np.dot(t, x.imag)
        if a > 0:
            return -self._creep_yaw
        return self._creep_yaw

    def _transform_activations(self, a, state):
        # a: up, right, down, left, center
        # if state.pos[1] < self._bounds[0] + self._margin:  # avoid right
        #     a[1] = -float("inf")
        # if state.pos[1] > self._bounds[1] - self._margin:  # avoid left
        #     a[3] = -float("inf")
        # if state.pos[2] < self._bounds[2] + self._margin:  # avoid down
        #     a[2] = -float("inf")
        # if state.pos[2] > self._bounds[3] - self._margin:  # avoid up
        #     a[0] = -float("inf")

        vx, vz = 0, 0

        vy = self._speed_y
        index = a.argmax().item()
        if index == 0:  # up
            vz = self._speed_z
        elif index == 1:  # right
            vx = self._speed_x
        elif index == 2:  # down
            vz = -self._speed_z
        elif index == 3:  # left
            vx = -self._speed_x

        rq = np.quaternion(*state.att)
        yq = np.quaternion(0, 0, 1, 0)
        y = rq * yq * np.conjugate(rq)
        y = y.imag * np.array([1, 1, 0])
        y /= np.linalg.norm(y)
        xq = np.quaternion(0, 1, 0, 0)
        x = rq * xq * np.conjugate(rq)
        x = x.imag * np.array([1, 1, 0])
        x /= np.linalg.norm(x)
        vxy = vx * x + vy * y
        return [vxy[0], vxy[1], vz]
