import numpy as np
import data

def train(X, Y_, param_niter=100, param_delta=1e-2):
	'''
	Arguments:
		X: Samples, np.ndarray N*D
		Y_: Correct labels, np.ndarray N*1
	
	Return values:
		W: Weights of logistic regression, np.ndarray D*C
		b: Biases of logistic regression, np.ndarray 1*C
	'''
	N = X.shape[0]
	D = X.shape[1]
	C = np.max(Y_) + 1
	W = np.random.normal(size=(D, C))
	b = np.random.normal(size=C)
	Y_onehot = np.zeros((N, C), dtype=np.int64)
	for i in range(N):
		Y_onehot[i, Y_[i]] = 1
	for i in range(param_niter):
		scores = X @ W + b
		expscores = np.exp(scores - np.max(scores, axis=1).reshape(N, 1))

		sumexp = np.sum(expscores, axis=1).reshape(N, 1)

		probs = expscores / sumexp
		logprobs = np.log(probs)

		loss = 0.0
		for j in range(N):
			loss += (-logprobs[j][Y_[j]])
		
		if i % 10 == 0:
			print(f"iteration: {i}, loss: {loss}")
		
		dL_ds = probs - Y_onehot
		grad_W = (1 / N * dL_ds.T @ X).T
		grad_b = np.average(dL_ds, axis=0)

		W += -param_delta * grad_W
		b += -param_delta * grad_b
	return W, b

def classify(X, W, b):
	N = X.shape[0]
	scores = X @ W + b
	expscores = np.exp(scores - np.max(scores, axis=1).reshape(N, 1))
	sumexp = np.sum(expscores, axis=1).reshape(N, 1)
	probs = expscores / sumexp
	return probs

def graph_classify(W, b):
	def c(X):
		probs = classify(X, W, b)
		return np.max(probs, axis=1)
	return c

if __name__ == "__main__":
	np.random.seed(100)

	D = 2
	K = 3
	C = 3
	M = 50
	X, Y_ = data.sample_gmm_2d(K, C, M)
	Yoh_ = np.zeros((Y_.shape[0], C))
	for i in range(Y_.shape[0]):
		Yoh_[i, Y_[i]] = 1

	W, b = train(X, Y_, 10000, 1e-1)

	probs = classify(X, W, b)
	Y = np.argmax(probs, axis=1)

	accuracy, confusion, precision, recall = data.eval_perf_multi(Y, Y_)

	print(f"{accuracy=}, {precision=}, {recall=}")
	print(f"confusion:\n{confusion}")

	data.graph_surface(graph_classify(W, b), [np.min(X, axis=0), np.max(X, axis=0)], 0.5, 1e-2, 1e-2)
	data.graph(X, Y, Y_)
