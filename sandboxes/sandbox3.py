import pickle
import matplotlib.pyplot as plt


fn = "weights2.pickle"

WIDTH = 16
HEIGHT = 16

with open(fn, "rb") as f:
    w = pickle.load(f)

img = w.cpu()[4, :].reshape(HEIGHT, WIDTH)
mi = img.min().item()
ma = img.max().item()
plt.imshow(-img, cmap="Greys")
plt.show()
