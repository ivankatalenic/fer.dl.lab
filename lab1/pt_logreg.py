import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import data

class PTLogreg(nn.Module):
	def __init__(self, D: int, C: int):
		super().__init__()
		self.W = nn.Parameter(torch.randn((C, D), dtype=torch.double))
		self.b = nn.Parameter(torch.randn((C,), dtype=torch.double))
	def forward(self, X: torch.Tensor) -> torch.Tensor:
		return torch.softmax(torch.mm(X, self.W.T) + self.b, 1)
	def get_loss(self, X: torch.Tensor, Yoh_: torch.Tensor) -> torch.Tensor:
		log = torch.log_softmax(torch.mm(X, self.W.T) + self.b, 1)
		return torch.sum(-Yoh_ * log) / X.size(0)

def train(model: PTLogreg, X: torch.Tensor, Yoh_: torch.Tensor, param_niter: int, param_delta: int, param_lambda: float = 0):
	optimizer = optim.SGD(model.parameters(), lr=param_delta)
	# optimizer = optim.SGD(model.parameters(), lr=param_delta, weight_decay=param_lambda)

	for i in range(param_niter):
		optimizer.zero_grad()

		loss = model.get_loss(X, Yoh_)
		# loss += model.W.pow(2).sum().sqrt() * param_lambda
		
		loss.backward()
		optimizer.step()

		if i % 10 == 0:
			print(f'{i=}, {loss=}')

def eval(model: PTLogreg, X: np.ndarray) -> np.ndarray:	
	predicted = model.forward(torch.as_tensor(X))
	return predicted.numpy(force=True)

def graph_classify(model: PTLogreg):
	def c(X: np.ndarray):
		probs = model.forward(torch.as_tensor(X)).numpy(force=True)
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

	model = PTLogreg(D, C)

	train(model, torch.as_tensor(X), torch.as_tensor(Yoh_), 10000, 1e-1, 1e-1)

	probs = eval(model, X)
	Y = np.argmax(probs, axis=1)

	accuracy, confusion, precision, recall = data.eval_perf_multi(Y, Y_)

	print(f"{accuracy=}, {precision=}, {recall=}")
	print(f"confusion:\n{confusion}")

	data.graph_surface(graph_classify(model), [np.min(X, axis=0), np.max(X, axis=0)], 0.5, 1e-2, 1e-2)
	data.graph(X, Y, Y_)
