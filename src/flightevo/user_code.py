import torch
import pickle
from pathlib import Path
from utils import AgileCommand

SPEED_X = 3.
SPEED_Y = 1.2
SPEED_Z = 1.2
RES_WIDTH = 16
RES_HEIGHT = 16
BOUNDS = [-10, 10, 0, 10]
GAMMA = 2.2
ACC = .4
DEVICE = "cpu"
MARGIN = .8

with open(Path(__file__).parent / "weights.pickle", "rb") as f:
    WEIGHTS = pickle.load(f).to(device=DEVICE)


def compute_command_vision_based(state, img):
    # s = self._transform_state(state)
    i = _transform_img(img, state)
    a = _activate(i)
    v = _transform_activations(a, state)
    c = AgileCommand(2)
    c.t = state.t
    c.velocity = v
    return c


def _activate(inputs):
    with torch.no_grad():
        x = torch.as_tensor(
            inputs, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        return WEIGHTS.mm(x).squeeze(1)


def _transform_activations(a, state):
    # a: up, right, down, left, center
    if state.pos[1] < BOUNDS[0] + MARGIN:  # avoid right
        a[1] = -float("inf")
    if state.pos[1] > BOUNDS[1] - MARGIN:  # avoid left
        a[3] = -float("inf")
    if state.pos[2] < BOUNDS[2] + MARGIN:  # avoid down
        a[2] = -float("inf")
    if state.pos[2] > BOUNDS[3] - MARGIN:  # avoid up
        a[0] = -float("inf")

    vy, vz = 0, 0
    vx = min(SPEED_X, state.vel[0] + ACC)
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
