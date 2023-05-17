import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

class Random2DGaussian:
	minx = -5
	maxx = 5
	miny = -5
	maxy = 5

	def create_random_parameters(self):
		x = (self.maxx - self.minx) * np.random.random_sample() + self.minx
		y = (self.maxy - self.miny) * np.random.random_sample() + self.miny
		eigvalx = (np.random.random_sample() * (self.maxx - self.minx) / 5)**2
		eigvaly = (np.random.random_sample() * (self.maxy - self.miny) / 5)**2
		eig = np.diag([eigvalx, eigvaly])
		phi = np.random.random_sample() * 2 * np.pi
		rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
		cov = rot.T @ eig @ rot
		return np.array([x, y]), cov
	
	def __init__(self, median=None, covariance=None):
		if median is None and covariance is None:
			self.median, self.covariance = self.create_random_parameters()
			return
		if type(median) is not np.ndarray:
			raise TypeError("the median is not of numpy.ndarray type")
		if median.shape != (2,):
			raise ValueError(f"the median doesn't have a compatible shape: required (2,), received {median.shape}")
		if type(covariance) is not np.ndarray:
			raise TypeError("the covariance is not of numpy.ndarray type")
		if covariance.shape != (2,2):
			raise ValueError(f"the covariance doesn't have a compatible shape: required (2,2), received {covariance.shape}")
		self.median = median.copy()
		self.covariance = covariance.copy()
	
	def get_sample(self, n):
		# rng = np.random.default_rng()
		# return rng.multivariate_normal(self.median, self.covariance, n, 'raise')
		return np.random.multivariate_normal(self.median, self.covariance, n, 'raise')

def sample_gauss_2d(C, N):
	'''
	Samples N data points from C random bivariate Gauss distributions.
	'''
	samples = np.empty((N * C, 2))
	labels = np.empty(N * C, dtype=np.int64)
	for i in range(C):
		dist = Random2DGaussian()
		samples[i * N : (i + 1) * N] = dist.get_sample(N)
		labels[i * N : (i + 1) * N] = i
	return samples, labels

def sample_gmm_2d(K, C, N):
	'''
	Samples N data points from K random bivariate Gauss distributions that are mapped to C (C <= K) labels.
	'''
	indices = np.hstack([np.arange(C), np.random.choice(C, K - C)])
	samples = np.empty((K * N, 2))
	labels = np.empty(K * N, dtype=np.int64)
	for i in range(K):
		dist = Random2DGaussian()
		samples[i * N : (i + 1) * N] = dist.get_sample(N)
		labels[i * N : (i + 1) * N] = indices[i]
	return samples, labels

def eval_perf_binary(Y, Y_):
	'''
	Evaluates the predicted and correct labels. It calculates accuracy, precision, and recall.
	'''
	TP = np.sum(Y[Y_ == 1])
	FP = np.sum(Y[Y_ == 0])
	TN = np.sum((1 - Y)[Y_ == 0])
	FN = np.sum((1 - Y)[Y_ == 1])
	accuracy = (TP + TN) / (TP + FP + TN + FN)
	precision = TP / (TP + FP)
	recall = TP / (TP + FN)
	return accuracy, precision, recall

def eval_perf_multi(Y, Y_):
	C = np.max(Y_) + 1
	
	confusion = np.zeros((C, C))
	for i in range(C):
		observed = Y[Y_ == i]
		for j in range(C):
			confusion[i, j] = np.count_nonzero(observed == j)
	accuracy = np.sum(np.diag(confusion)) / np.sum(confusion)
	precision = np.diag(confusion) / np.sum(confusion, axis=0)
	recall = np.diag(confusion) / np.sum(confusion, axis=1)
	return accuracy, confusion, precision, recall

def eval_AP(Y_r):
	'''
	Calculates average precision.
	
	Arguments:
		Y_r: List of correct labels sorted by their posterior probabilities p(c_1|x).
	'''
	sum = np.sum(Y_r)
	total = Y_r.shape[0]
	AP = 0.0
	for i in range(total):
		TP = np.sum(Y_r[i:])
		P = total - i
		FP = P - TP
		precision = TP / (TP + FP)
		AP += (precision * Y_r[i]) / sum
	return AP

def graph(X, Y, Y_):
	'''
	Graphs the classification data.
	
	Arguments:
		X: Samples (np.ndarray, N*2)
		Y: Predicted labels (np.ndarray, N)
		Y_: Correct labels (np.ndarray, N)
	'''
	C = np.max(Y_) + 1
	cmap = mpl.colormaps["gist_rainbow"]
	for c in range(C):
		color = cmap(c / (C - 1))
		correct = X[np.logical_and(Y_ == Y, Y_ == c)]
		plt.scatter(correct[:, 0], correct[:, 1], color=color, marker='o', edgecolor='black')
		incorrect = X[np.logical_and(Y_ != Y, Y_ == c)]
		plt.scatter(incorrect[:, 0], incorrect[:, 1], color=color, marker='s', edgecolor='black')
	plt.show()


def graph_surface(fun, rect, offset, width, height, decision_width=0.05):
	total_width = rect[1][0] - rect[0][0]
	total_height = rect[1][1] - rect[0][1]

	x = np.linspace(rect[0][0], rect[1][0], round(total_width / width))
	y = np.linspace(rect[0][1], rect[1][1], round(total_height / height))
	xx, yy = np.meshgrid(x, y)
	samples = np.stack([xx.flatten(), yy.flatten()], axis=-1)
	values = fun(samples)
	values = values.reshape((y.size, x.size))
	delta = np.amax(np.abs(values - offset))
	plt.pcolormesh(xx, yy, values, vmin=(offset - delta), vmax=(offset + delta))
	values[np.abs(values - offset) > decision_width] = 0
	plt.contour(xx, yy, values, levels=1, colors='k')

if __name__ == '__main__':
	y = np.array([1, 0, 1, 0, 1, 0])
	print(y)
	print(eval_AP(y))
