from flightevo.dodger import Dodger
from flightevo import utils


class Bencher(Dodger):
    def __init__(self, resolution_width, resolution_height,
                 speed_x, speed_y, speed_z, bounds, gamma, acc, margin,
                 creep_z, creep_yaw):
        self._creep_z = creep_z
        self._creep_yaw = creep_yaw
        self._target_x, self._target_y, self._target_z = 0, 0, 0
        super().__init__(resolution_width, resolution_height,
                         speed_x, speed_y, speed_z, bounds, gamma, acc, margin)

    def set_target(self, x, y, z):
        self._target_x, self._target_y, self._target_z = x, y, z

    def compute_command_vision_based(self, state, img):
        # s = self._transform_state(state)
        i = self._transform_img(img, state)
        a = self._mlp.activate(i)
        v = self._transform_activations(a, state)
        v = self._adjust_height(v, state)
        yawrate = self._adjust_yaw(state)
        c = utils.AgileCommand(2)
        c.t = state.t
        c.velocity = v
        c.yawrate = yawrate
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
        vx, vy, vz = utils.translate(vx, vy, vz, *state.att)
        return [vx, vy, vz]
