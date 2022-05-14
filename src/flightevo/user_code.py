import torch
import pickle
from utils import AgileCommand

SPEED_X = 2.0
SPEED_Y = .8
SPEED_Z = .8
RES_WIDTH = 40
RES_HEIGHT = 30
BOUNDS = [-10, 10, 0, 10]
GAMMA = 2.2

with open("weights.pickle", "rb") as f:
    WEIGHTS = pickle.load(f)


def compute_command_vision_based(self, state, img):
    # s = self._transform_state(state)
    i = _transform_img(img, state)
    a = self._mlp.activate(i)
    v = _transform_activations(a, state)
    return AgileCommand(
        t=state.t, mode=2, yawrate=0, velocity=v)


def activate(inputs):
    with torch.no_grad():
        x = torch.as_tensor(
            inputs, dtype=torch.float32, device="cuda").unsqueeze(1)
        return WEIGHTS.mm(x).squeeze(1)


def _transform_activations(a, state):
    vy, vz = 0, 0
    vx = min(SPEED_X, state.vel[0] + .2)
    index = a.argmax().item()
    if index == 0:  # up
        vz = SPEED_Z
    elif index == 1:  # right
        vy = -SPEED_Y
    elif index == 2:  # down
        vz = -SPEED_Z
    elif index == 3:  # left
        vy = SPEED_Y
    return [vx, vy, vz]


def _transform_img(img, state):
    r, c = img.shape
    k0 = int(r / RES_HEIGHT)
    k1 = int(c / RES_WIDTH)
    # copy needed due to non-writeable nparray
    new_img = 1 - torch.tensor(img) \
        .unfold(0, k0, k0).unfold(1, k1, k1).amin((-1, -2),)
    # add border
    right = max(state.pos[1] - BOUNDS[0] - 1, .0)
    left = max(BOUNDS[1] - state.pos[1] - 1, .0)
    down = max(state.pos[2] - BOUNDS[2] - 1, .0)
    up = max(BOUNDS[3] - state.pos[2] - 1, .0)
    bw = int(RES_WIDTH / 4)
    bh = int(RES_HEIGHT / 4)
    new_img[:bh, :].clamp_(1 - up / 100)
    new_img[-bh:, :].clamp_(1 - down / 100)
    new_img[:, :bw].clamp_(1 - left / 100)
    new_img[:, -bw:].clamp_(1 - right / 100)
    # non-linear scaling
    new_img.pow_(GAMMA)
    return new_img.view(-1)
