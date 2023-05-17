import numpy as np

def get_loss(X, Y_, w, b):
	scores = X @ w + b
	scores_exp = np.exp(scores)
	probs = scores_exp / (1.0 + scores_exp)
	conf = probs
	conf[Y_ == 0] = 1 - conf[Y_ == 0]
	loss = np.average(-np.log(conf))
	return loss

def train(X, Y_, param_niter=100, param_delta=1e-2):
	'''
	Arguments:
		X: Samples, np.ndarray N*D
		Y_: Correct labels, np.ndarray N*1
	
	Return values:
		w, b: Parameters of binary logistic regression
	'''
	# rng = np.random.default_rng()
	# w = rng.normal(size=X.shape[1])
	w = np.random.normal(size=X.shape[1])
	b = 0.0

	for i in range(param_niter):
		scores = X @ w + b
		scores_exp = np.exp(scores)
		probs = scores_exp / (1.0 + scores_exp)

		dL_dscores = probs - Y_
		N = X.shape[0]
		grad_w = 1 / N * (dL_dscores @ X)
		grad_b = np.average(dL_dscores, 0)

		# Gradient checking
		# epsilon = 1e-4
		# grad_w_approx = np.empty(X.shape[1])
		# for j in range(X.shape[1]):
		# 	w_delta = np.zeros(X.shape[1])
		# 	w_delta[j] = 1
		# 	grad_w_approx[j] = (get_loss(X, Y_, w + w_delta * epsilon, b) - get_loss(X, Y_, w + w_delta * (-epsilon), b)) / (2 * epsilon)
		# grad_b_approx = (get_loss(X, Y_, w, b + epsilon) - get_loss(X, Y_, w, b - epsilon)) / (2 * epsilon)

		sigma = probs
		sigma[Y_ == 0] = 1 - sigma[Y_ == 0]
		loss = np.sum(-np.log(sigma))
		
		if i % 10 == 0:
			print(f"iteration: {i}, loss: {loss}")
			# print(f"gradient diffs: w={grad_w - grad_w_approx}, b={np.abs(grad_b - grad_b_approx)}")
		
		w += -param_delta * grad_w
		b += -param_delta * grad_b
	
	return w, b

def classify(X, w, b):
	'''
	Arguments:
		X: Data, np.ndarray N*D
		w, b: Parameters of binary logistic regression

	Return values:
		probs: Probabilities for class C1 for each sample
	'''
	scores = X @ w + b
	scores_exp = np.exp(scores)
	probs = scores_exp / (1.0 + scores_exp)
	return probs
