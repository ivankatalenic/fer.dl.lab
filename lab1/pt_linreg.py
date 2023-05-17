import torch
import torch.nn as nn
import torch.optim as optim

# Define the parameters of the model (the leaf nodes of the backward-pass graph)
# Assign random values to them.
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Define dataset
X = torch.tensor([1, 2, 3, 4,  5,  6,  7,  8,  9])
Y = torch.tensor([3, 5, 7, 9, 11, 13, 15, 17, 19])

optimizer = optim.SGD([a, b], lr=1e-2)

for i in range(100):
	# Define the model's output
	Y_ = a*X + b

	diff = (Y - Y_)
	loss = torch.sum(diff**2)

	# Calculate the parameters' gradients
	loss.backward()

	a.grad /= len(X)
	b.grad /= len(X)

	# Move the parameters using the computed gradients
	optimizer.step()

	print(f'{i=}, {loss=}, {a=}, {b=}')

	with torch.no_grad():
		a_man_grad = -2 * torch.sum(diff * X) / len(X)
		b_man_grad = -2 * torch.sum(diff) / len(X)
		print(f'grad_diff={[a.grad - a_man_grad, b.grad - b_man_grad]}')

	# Reset the gradients to zero
	optimizer.zero_grad()
