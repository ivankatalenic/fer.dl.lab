import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import data

class PTDeep(nn.Module):
	def __init__(self, layers, activation, device=None):
		super().__init__()
		self.weights = nn.ParameterList()
		self.biases = nn.ParameterList()
		for i in range(1, len(layers)):
			self.weights.append(nn.Parameter(torch.randn((layers[i - 1], layers[i]), dtype=torch.float, device=device)))
			self.biases.append(nn.Parameter(torch.randn((layers[i],), dtype=torch.float, device=device)))
		self.activation = activation
	def forward(self, X: torch.Tensor) -> torch.Tensor:
		length = len(self.weights)
		h = X
		for i in range(length):
			h = torch.mm(h, self.weights[i]) + self.biases[i]
			if i < length - 1:
				h = self.activation(h)
		return torch.softmax(h, 1)
	def get_loss(self, X: torch.Tensor, Yoh_: torch.Tensor) -> torch.Tensor:
		length = len(self.weights)
		h = X
		for i in range(length):
			h = torch.mm(h, self.weights[i]) + self.biases[i]
			if i < length - 1:
				h = self.activation(h)
		log = torch.log_softmax(h, 1)
		return torch.sum(-Yoh_ * log) / X.size(0)

def train(
	model: PTDeep,
	X: torch.Tensor,
	Yoh_: torch.Tensor,
	param_niter: int,
	param_lr: int,
	param_reg: float = 0,
	print_debug: bool = False,
	callback = None
):
	optimizer = optim.SGD(model.parameters(), lr=param_lr, weight_decay=param_reg)
	losses = np.empty((param_niter,))

	for i in range(param_niter):
		optimizer.zero_grad()

		loss = model.get_loss(X, Yoh_)
		losses[i] = loss.item()
		if callback is not None:
			callback(i)
		
		loss.backward()
		optimizer.step()

		if i % 1000 == 0 and print_debug:
			print(f'{i=}, loss={loss.item()}')
	return losses

def train_mb(
	model: PTDeep,
	X: torch.Tensor,
	Yoh_: torch.Tensor,
	batch_size: int,
	niter: int,
	lr: int,
	reg: float = 0,
	shuffle: bool = True,
	print_debug: bool = False,
	callback = None
):
	batch_size = len(X) if batch_size <= 0 else batch_size

	optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=reg)
	losses = np.empty(niter)
	for i in range(niter):
		loss = model.get_loss(X, Yoh_)
		losses[i] = loss.item()

		X_pool = X
		Yoh_pool = Yoh_

		if shuffle:
			shuffle_indices = torch.randperm(len(X))
			X_pool = X[shuffle_indices]
			Yoh_pool = Yoh_[shuffle_indices]
		
		for b_start in range(0, len(X_pool), batch_size):
			b_end = b_start + batch_size if b_start + batch_size <= len(X_pool) else len(X_pool)
			
			X_batch = X_pool[b_start : b_end]
			Yoh_batch = Yoh_pool[b_start : b_end]
			
			optimizer.zero_grad()

			loss = model.get_loss(X_batch, Yoh_batch)
			if callback is not None:
				callback(i)
			
			loss.backward()
			optimizer.step()

		if print_debug:
			print(f'{i=}, loss={losses[i]}')
	return losses

def train_mb_adam(
	model: PTDeep,
	X: torch.Tensor,
	Yoh_: torch.Tensor,
	batch_size: int,
	niter: int,
	lr: int,
	reg: float = 0,
	shuffle: bool = True,
	print_debug: bool = False,
	callback = None
):
	batch_size = len(X) if batch_size <= 0 else batch_size

	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
	losses = np.empty(niter)
	for i in range(niter):
		loss = model.get_loss(X, Yoh_)
		losses[i] = loss.item()

		X_pool = X
		Yoh_pool = Yoh_

		if shuffle:
			shuffle_indices = torch.randperm(len(X))
			X_pool = X[shuffle_indices]
			Yoh_pool = Yoh_[shuffle_indices]
		
		for b_start in range(0, len(X_pool), batch_size):
			b_end = b_start + batch_size if b_start + batch_size <= len(X_pool) else len(X_pool)
			
			X_batch = X_pool[b_start : b_end]
			Yoh_batch = Yoh_pool[b_start : b_end]
			
			optimizer.zero_grad()

			loss = model.get_loss(X_batch, Yoh_batch)
			if callback is not None:
				callback(i)
			
			loss.backward()
			optimizer.step()

		if print_debug:
			print(f'{i=}, loss={losses[i]}')
	return losses

def train_mb_adam_adapt(
	model: PTDeep,
	X: torch.Tensor,
	Yoh_: torch.Tensor,
	batch_size: int,
	niter: int,
	lr: int,
	reg: float = 0,
	shuffle: bool = True,
	print_debug: bool = False,
	gamma: float = 1-1e-4,
	callback = None
):
	batch_size = len(X) if batch_size <= 0 else batch_size

	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
	losses = np.empty(niter)
	for i in range(niter):
		loss = model.get_loss(X, Yoh_)
		losses[i] = loss.item()

		X_pool = X
		Yoh_pool = Yoh_

		if shuffle:
			shuffle_indices = torch.randperm(len(X))
			X_pool = X[shuffle_indices]
			Yoh_pool = Yoh_[shuffle_indices]
		
		for b_start in range(0, len(X_pool), batch_size):
			b_end = b_start + batch_size if b_start + batch_size <= len(X_pool) else len(X_pool)
			
			X_batch = X_pool[b_start : b_end]
			Yoh_batch = Yoh_pool[b_start : b_end]
			
			optimizer.zero_grad()

			loss = model.get_loss(X_batch, Yoh_batch)
			if callback is not None:
				callback(i)
			
			loss.backward()
			optimizer.step()

		scheduler.step()

		if print_debug:
			print(f'{i=}, loss={losses[i]}')
	return losses

def eval(model: PTDeep, X: np.ndarray) -> np.ndarray:
	'''
	Returns class probabilities for each sample.
	'''
	predicted = model.forward(torch.as_tensor(X))
	return predicted.numpy(force=True)

def graph_classify(model: PTDeep):
	def c(X: np.ndarray):
		probs = model.forward(torch.as_tensor(X)).numpy(force=True)
		return np.max(probs, axis=1)
	return c

def count_params(model: PTDeep) -> int:
	total = 0
	for name, param in model.named_parameters():
		print(f'{name=}, size: {param.size()}')
		total += param.numel()
	print(f'total number of parameters: {total}')

if __name__ == "__main__":
	np.random.seed(100)

	D = 2
	K = 3
	C = 2
	M = 10
	X, Y_ = data.sample_gmm_2d(K, C, M)
	Yoh_ = np.zeros((Y_.shape[0], C), dtype=np.float32)
	for i in range(Y_.shape[0]):
		Yoh_[i, Y_[i]] = 1

	model = PTDeep([D, 10, 10, C], torch.sigmoid, device='cuda')

	train(model, torch.as_tensor(X, device='cuda'), torch.as_tensor(Yoh_, device='cuda'), 1000, 1e-2, 1e-3, True)

	probs = eval(model, X)
	Y = np.argmax(probs, axis=1)

	accuracy, confusion, precision, recall = data.eval_perf_multi(Y, Y_)
	ap = data.eval_AP(Y_[np.argsort(probs[:, 1])])
	print(f"{accuracy=}, {precision=}, {recall=}, {ap=}")
	print(f"confusion:\n{confusion}")
	count_params(model)

	data.graph_surface(graph_classify(model), [np.min(X, axis=0), np.max(X, axis=0)], 0.5, 1e-2, 1e-2)
	data.graph(X, Y, Y_)
