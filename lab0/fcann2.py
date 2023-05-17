import numpy as np

def train(X, Y_, hidden_nodes=10, param_niter=1e5, param_delta=1e-2):
	N = X.shape[0]
	D = X.shape[1]
	C = np.max(Y_) + 1

	X_norm = (X - np.average(X, axis=0)) / np.std(X, axis=0)
	Y_onehot = np.zeros((N, C))
	for i in range(N):
		Y_onehot[i, Y_[i]] = 1

	W1 = np.random.normal(size=(hidden_nodes, D))
	b1 = np.random.normal(size=(hidden_nodes, 1))
	W2 = np.random.normal(size=(C, hidden_nodes))
	b2 = np.random.normal(size=(C, 1))

	for i in range(param_niter):
		# Compute layer value (forward pass)

		# 1st linear transformation
		S1 = X_norm @ W1.T + b1.T
		# ReLU transformation
		H1 = np.clip(S1, a_min=0.0, a_max=None)
		# 2nd linear transformation
		S2 = H1 @ W2.T + b2.T
		# Softmax
		S2_exp_adj = np.exp(S2 - np.max(S2, axis=1).reshape(N, 1))
		probs = S2_exp_adj / np.sum(S2_exp_adj, axis=1).reshape(N, 1)

		# Compute loss

		log_probs = np.log(probs)
		loss = 0.0
		for j in range(N):
			loss -= log_probs[j, Y_[j]]
		
		if i % 10 == 0:
			print(f"iteration: {i}, loss: {loss}")

		# Compute layer gradients (backward pass)

		G_S2 = probs - Y_onehot
		G_W2 = G_S2.T @ H1
		G_b2 = np.sum(G_S2, axis=0).reshape(1, -1)
		G_H1 = G_S2 @ W2
		G_S1 = G_H1 * H1
		G_W1 = G_S1.T @ X_norm
		G_b1 = np.sum(G_S1, axis=0).reshape(1, -1)

		# Adjust the parameters

		W1 += -param_delta * 1/N * G_W1
		b1 += -param_delta * 1/N * G_b1.T
		W2 += -param_delta * 1/N * G_W2
		b2 += -param_delta * 1/N * G_b2.T

	return W1, b1, W2, b2

def classify(X, W1, b1, W2, b2):
	N = X.shape[0]

	X_norm = (X - np.average(X, axis=0)) / np.std(X, axis=0)

	# 1st linear transformation
	S1 = X_norm @ W1.T + b1.T
	# ReLU transformation
	H1 = np.clip(S1, a_min=0.0, a_max=None)
	# 2nd linear transformation
	S2 = H1 @ W2.T + b2.T
	# Softmax
	S2_exp_adj = np.exp(S2 - np.max(S2, axis=1).reshape(N, 1))
	probs = S2_exp_adj / np.sum(S2_exp_adj, axis=1).reshape(N, 1)

	return probs
