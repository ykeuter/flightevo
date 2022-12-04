import numpy as np
import torch
import timeit

width = 320
height = 240
dev_trans = "cuda"
dev_act = "cpu"

img = np.random.rand(240, 320)
w = torch.rand(5, height * width, device=dev_act)


def fun():
    r, c = img.shape
    k0 = int(r / height)
    k1 = int(c / width)
    # copy needed due to non-writeable nparray
    new_img = 1 - torch.tensor(img, device=dev_trans) \
        .unfold(0, k0, k0).unfold(1, k1, k1).amin((-1, -2),)

    # non-linear scaling
    new_img.pow_(2.2)

    inputs = new_img.view(-1)

    with torch.no_grad():
        x = torch.as_tensor(
            inputs, dtype=torch.float32, device=dev_act).unsqueeze(1)
        x = w.mm(x)
    output = x.squeeze(1)


print(timeit.timeit(fun, number=10000))
