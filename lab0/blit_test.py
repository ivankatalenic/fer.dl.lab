import matplotlib.pyplot as plt
import numpy as np

from blitmanager import BlitManager

x = np.linspace(-np.pi, np.pi, 200)
# make a new figure
fig, ax = plt.subplots()
# add a line
(ln,) = ax.plot(x, np.sin(x), marker='o', ls='', animated=True)
# add a frame number
fr_number = ax.annotate(
    "0",
    (0, 1),
    xycoords="axes fraction",
    xytext=(10, -10),
    textcoords="offset points",
    ha="left",
    va="top",
    animated=True,
)
bm = BlitManager(fig.canvas, [ln, fr_number])
# make sure our window is on the screen and drawn
plt.show(block=False)
plt.pause(.1)

for j in range(10000):
    # update the artists
	y = np.sin(x + (j / 1000) * np.pi)
    ln.set_ydata(y)
    fr_number.set_text("frame: {j}".format(j=j))
    # tell the blitting manager to do its thing
    bm.update()
