import torch
import numpy as np
import matplotlib.pyplot as plt

import data
import binlogreg
import logreg
import fcann2

def dummy_classifier(X):
	return X[:, 0] + X[:, 1] - 5

def binary_classification():
	X, Y_ = data.sample_gauss_2d(2, 100)

	w, b = binlogreg.train(X, Y_, 100, 8)

	probs = binlogreg.classify(X, w, b)
	Y = np.around(probs)

	accuracy, precision, recall = data.eval_perf_binary(Y, Y_)
	AP = data.eval_AP(Y_[probs.argsort()])

	print(f"{accuracy=}, {precision=}, {recall=}, {AP=}")

	data.graph_surface(lambda X: binlogreg.classify(X, w, b), [np.min(X, axis=0), np.max(X, axis=0)], 0.5, 1e-2, 1e-2)
	data.graph(X, Y, Y_)

def logreg_classify(W, b):
	def c(X):
		probs = logreg.classify(X, W, b)
		return np.max(probs, axis=1)
	return c

def fcann2_classify(W1, b1, W2, b2):
	def c(X):
		probs = fcann2.classify(X, W1, b1, W2, b2)
		return np.max(probs, axis=1)
	return c
	
def classification():
	X, Y_ = data.sample_gauss_2d(3, 100)

	W, b = logreg.train(X, Y_, 1000, 10)

	probs = logreg.classify(X, W, b)
	Y = np.argmax(probs, axis=1)

	accuracy, confusion, precision, recall = data.eval_perf_multi(Y, Y_)

	print(f"{accuracy=}, {precision=}, {recall=}")
	print(f"confusion:\n{confusion}")

	data.graph_surface(logreg_classify(W, b), [np.min(X, axis=0), np.max(X, axis=0)], 0.5, 1e-2, 1e-2)
	data.graph(X, Y, Y_)

def classification2():
	X, Y_ = data.sample_gauss_2d(3, 100)

	logreg.train(X, Y_, 1000, 10)
	X = np.array([1, 2, 3])

def multilayer_classification():
	X, Y_ = data.sample_gmm_2d(5, 2, 50)

	W1, b1, W2, b2 = fcann2.train(X, Y_, 10, 500000, 0.03)

	probs = fcann2.classify(X, W1, b1, W2, b2)
	Y = np.argmax(probs, axis=1)

	accuracy, confusion, precision, recall = data.eval_perf_multi(Y, Y_)

	print(f"{accuracy=}, {precision=}, {recall=}")
	print(f"confusion:\n{confusion}")

	data.graph_surface(fcann2_classify(W1, b1, W2, b2), [np.min(X, axis=0), np.max(X, axis=0)], 0.25, 1e-2, 1e-2)
	data.graph(X, Y, Y_)

if __name__ == "__main__":
	seed = np.random.randint(0, 1000000)
	seed = 233226
	np.random.seed(seed)

	# binary_classification()
	# classification()
	# classification2()

	# samples, labels = data.sample_gmm_2d(4, 2, 50)
	# data.graph(samples, labels, labels)

	multilayer_classification()
	print(f"seed: {seed}")
