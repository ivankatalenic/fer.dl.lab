import matplotlib.pyplot as plt
import numpy as np

from blitmanager import BlitManager
import logreg

def logreg_classify(W, b):
	def c(X):
		probs = logreg.classify(X, W, b)
		return np.max(probs, axis=1)
	return c

class ChangeDrawer:
	def __init__(self, X, Y_):
		self.X = X
		self.Y_ = Y_
		self.mesh_artist = None

		fig, ax = plt.subplots()
		self.ax = ax

		self.graph_surface(lambda X: np.ones((X.shape[0], 1)), [np.min(X, axis=0), np.max(X, axis=0)], 0.5, 1e-2, 1e-2)
		
		self.bm = BlitManager(fig.canvas, [self.mesh_artist])

		plt.show(block=False)
		plt.pause(.1)

	def update(self, W, b):
		self.graph_surface(logreg_classify(W, b), [np.min(self.X, axis=0), np.max(self.X, axis=0)], 0.5, 1e-2, 1e-2)

		self.bm.update()
	
	def graph_surface(self, fun, rect, offset, width, height, decision_width=0.05):
		total_width = rect[1][0] - rect[0][0]
		total_height = rect[1][1] - rect[0][1]
		x = np.linspace(rect[0][0], rect[1][0], round(total_width / width))
		y = np.linspace(rect[0][1], rect[1][1], round(total_height / height))
		xx, yy = np.meshgrid(x, y)
		mesh_grid = np.stack([xx.flatten(), yy.flatten()], axis=-1)
		
		values = fun(mesh_grid)
		values = values.reshape((y.size, x.size))
		delta = np.amax(np.abs(values - offset))
		vmin = offset - delta
		vmax = offset + delta

		if self.mesh_artist is None:
			self.mesh_artist = self.ax.pcolormesh(xx, yy, values, vmin=vmin, vmax=vmax, animated=True)
		else:
			self.mesh_artist.set_array(values)
			self.mesh_artist.set_clim(vmin, vmax)
		
		# values[np.abs(values - offset) > decision_width] = 0
		# plt.contour(xx, yy, values, levels=1, colors='k')
