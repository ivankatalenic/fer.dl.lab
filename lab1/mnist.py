import copy

import numpy as np
import torch
import torchvision
import sklearn.svm as svm

import matplotlib.pyplot as plt

import pt_deep
import data

def get_data(device='cuda'):
	dataset_root = 'D:/tmp/mnist'
	mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=False)
	mnist_test  = torchvision.datasets.MNIST(dataset_root, train=False, download=False)
	
	x_train, y_train = mnist_train.data.to(device=device), mnist_train.targets.to(device=device)
	x_test,  y_test  = mnist_test.data.to(device=device), mnist_test.targets.to(device=device)
	x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)
	
	N_train = x_train.shape[0]
	N_test  = x_test.shape[0]
	D = x_train.shape[1] * x_train.shape[2]
	C = y_train.max().add_(1).item()

	x_train_vec = x_train.reshape((N_train, D))
	x_test_vec  = x_test.reshape((N_test, D))

	y_train_oh = torch.zeros((N_train, C), dtype=torch.float32, device=device)
	for i in range(N_train):
		y_train_oh[i, y_train[i]] = 1.0
	y_test_oh = torch.zeros((N_test, C), dtype=torch.float32, device=device)
	for i in range(N_test):
		y_test_oh[i, y_test[i]] = 1.0
	
	return x_train_vec, y_train, y_train_oh, x_test_vec, y_test, y_test_oh

def dot1():
	x_train_vec, y_train, y_train_oh, x_test_vec, y_test, y_test_oh = get_data()

	layers = [784, 10]
	niter = 10000
	lr = 1
	regs = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 2]

	fig = plt.figure(layout='constrained')
	fig.suptitle(f'mnist: {layers=}, {niter=}, {lr=}')
	subfigs = fig.subfigures(len(regs) + 1, 1)
	for i, reg in enumerate(regs):

		print(f'{i+1}/{len(regs)}: {reg=}')
	
		model = pt_deep.PTDeep(layers, torch.relu, device='cuda')
		losses = pt_deep.train(model, x_train_vec, y_train_oh, niter, lr, reg)
		
		probs_train = pt_deep.eval(model, x_train_vec)
		y_train_p = np.argmax(probs_train, axis=1)
		acc_train, _, _, _ = data.eval_perf_multi(y_train.numpy(force=True), y_train_p)

		probs_test = pt_deep.eval(model, x_test_vec)
		y_test_p = np.argmax(probs_test, axis=1)
		acc_test, _, _, _ = data.eval_perf_multi(y_test.numpy(force=True), y_test_p)

		subfig = subfigs[i]
		subfig.suptitle(f'{reg=}, loss={round(losses[-1], 3)}, acc_train={round(acc_train, 3)}, acc_test={round(acc_test, 3)}')
		
		axs = subfig.subplots(nrows=1, ncols=10)
		weights = model.weights[0].numpy(force=True) # Only one hidden layer in this case
		for i in range(10):
			ax = axs[i]
			image = weights[:, i].reshape((28, 28))
			ax.axis('off')
			ax.imshow(image, cmap='gray', interpolation='none')
	
	subfig = subfigs[-1]
	subfig.suptitle(f'mnist digits')
	axs = subfig.subplots(nrows=1, ncols=10)
	x_train = x_train_vec.reshape((x_train_vec.shape[0], 28, 28))
	for i in range(10):
		ax = axs[i]
		image = (x_train[y_train == i])[0].numpy(force=True)
		ax.axis('off')
		ax.imshow(image, cmap='gray', interpolation='none')
	
	plt.show()

def dot2():
	x_train_vec, y_train, y_train_oh, x_test_vec, y_test, y_test_oh = get_data()
	
	# p_layers = [[784, 10], [784, 100, 10], [784, 100, 100, 10], [784, 100, 100, 100, 10]]
	p_layers = [[784, 10], [784, 20, 10], [784, 20, 20, 20, 10]]
	p_iter = 50000
	# p_lr = [2, 1, 0.5, 0.1, 0.01]
	p_lr = [2, 0.5, 0.01, 0.001]

	fig, axs = plt.subplots(nrows=len(p_lr), ncols=len(p_layers), tight_layout=True)
	for i in range(len(p_layers)):
		layers = p_layers[i]
		for j in range(len(p_lr)):
			lr = p_lr[j]

			print(f'{j + 1 + i * len(p_lr)}/{len(p_lr) * len(p_layers)}: {layers=}, {lr=}')

			model = pt_deep.PTDeep(layers, torch.relu, device='cuda')
			losses = pt_deep.train(model, x_train_vec, y_train_oh, p_iter, lr)
			
			probs_train = pt_deep.eval(model, x_train_vec)
			y_train_p = np.argmax(probs_train, axis=1)
			accuracy_train, confusion_train, precision_train, recall_train = data.eval_perf_multi(y_train.numpy(force=True), y_train_p)

			probs_test = pt_deep.eval(model, x_test_vec)
			y_test_p = np.argmax(probs_test, axis=1)
			accuracy_test, confusion_test, precision_test, recall_test = data.eval_perf_multi(y_test.numpy(force=True), y_test_p)

			ax = axs[j, i]
			ax.set_title(f'{str(layers)}, {lr=}\n tr_acc={round(accuracy_train, 3)}, te_acc={round(accuracy_test, 3)}, loss={round(losses[-1], 3)}', {'fontsize': 8})
			ax.plot(losses)
			ax.set_xlabel('iteration')
			ax.set_ylabel('loss')
	plt.show()

def dot3():
	x_train_vec, y_train, y_train_oh, x_test_vec, y_test, y_test_oh = get_data()
	layers = [784, 100, 100, 10]
	niter = 50000
	lr = 0.05
	regs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

	acc_train_l = np.empty(len(regs))
	acc_test_l = np.empty(len(regs))
	loss_l = np.empty(len(regs))

	for i in range(len(regs)):
		reg = regs[i]
		
		print(f'{i + 1}/{len(regs)}: {reg=}')

		model = pt_deep.PTDeep(layers, torch.relu, device='cuda')
		losses = pt_deep.train(model, x_train_vec, y_train_oh, niter, lr, reg)
		loss_l[i] = round(losses[-1], 3)
		
		probs_train = pt_deep.eval(model, x_train_vec)
		y_train_p = np.argmax(probs_train, axis=1)
		accuracy_train, _, _, _ = data.eval_perf_multi(y_train.numpy(force=True), y_train_p)
		acc_train_l[i] = round(accuracy_train, 3)

		probs_test = pt_deep.eval(model, x_test_vec)
		y_test_p = np.argmax(probs_test, axis=1)
		accuracy_test, _, _, _ = data.eval_perf_multi(y_test.numpy(force=True), y_test_p)
		acc_test_l[i] = round(accuracy_test, 3)

	fig, ax = plt.subplots(layout='constrained')
	fig.suptitle(f'mnist: {layers=}, {niter=}, {lr=}')
	values = [acc_train_l, acc_test_l]
	labels = ['Train', 'Test']
	width = 1
	ticks = np.linspace(0, (len(regs) - 1) * (len(labels) + 1) * width, len(regs))
	offset = 0
	for i in range(2):
		b = ax.bar(ticks + offset, values[i], width, label=labels[i])
		ax.bar_label(b)
		offset += width
	ax.set_ylabel('Accuracy')
	ax.set_xticks(ticks + width/2, [f'{reg=}\nloss={loss_l[i]}' for i, reg in enumerate(regs)])
	ax.legend()
	plt.show()

def dot4():
	x_train_vec, y_train, y_train_oh, x_test_vec, y_test, y_test_oh = get_data()
	threshold = int(0.2 * len(x_train_vec))
	x_val_vec = x_train_vec[:threshold]
	x_train_vec = x_train_vec[threshold:]
	y_val = y_train[:threshold]
	y_train = y_train[threshold:]
	y_val_oh = y_train_oh[:threshold]
	y_train_oh = y_train_oh[threshold:]

	layers = [784, 100, 100, 10]
	niter = 400_000
	lr = 0.01
	reg = 0

	model = pt_deep.PTDeep(layers, torch.relu, device='cuda')
	
	early_model = None
	early_acc = 0
	early_iter = 0
	acc_l = np.empty(niter)
	def callback(iter: int):
		nonlocal model, x_val_vec, y_val, early_model, early_acc, early_iter
		probs = pt_deep.eval(model, x_val_vec)
		classes = np.argmax(probs, axis=1)
		acc, _, _, _ = data.eval_perf_multi(y_val.numpy(force=True), classes)
		acc_l[iter] = acc
		if acc > early_acc:
			early_acc = acc
			early_model = copy.deepcopy(model)
			early_iter = iter
	
	losses = pt_deep.train(model, x_train_vec, y_train_oh, niter, lr, reg, True, callback)

	train_probs = pt_deep.eval(early_model, x_train_vec)
	y_train_pred = np.argmax(train_probs, axis=1)
	acc_train, _, _, _ = data.eval_perf_multi(y_train.numpy(force=True), y_train_pred)
	test_probs = pt_deep.eval(early_model, x_test_vec)
	y_test_pred = np.argmax(test_probs, axis=1)
	acc_test, _, _, _ = data.eval_perf_multi(y_test.numpy(force=True), y_test_pred)
	print(f'Early model ({early_iter}/{niter}): Acc Train = {round(acc_train, 3)}, Acc Test = {round(acc_test, 3)}')
	
	train_probs = pt_deep.eval(model, x_train_vec)
	y_train_pred = np.argmax(train_probs, axis=1)
	acc_train, _, _, _ = data.eval_perf_multi(y_train.numpy(force=True), y_train_pred)
	test_probs = pt_deep.eval(model, x_test_vec)
	y_test_pred = np.argmax(test_probs, axis=1)
	acc_test, _, _, _ = data.eval_perf_multi(y_test.numpy(force=True), y_test_pred)
	print(f'Final model: Acc Train = {round(acc_train, 3)}, Acc Test = {round(acc_test, 3)}')

	fig, ax = plt.subplots(layout='constrained')
	ax.set_title(f'mnist: {layers=}, {lr=}, {reg=}')
	ax.plot(acc_l, label='Validation')
	ax.annotate(
		'Early model',
		xy=(early_iter, acc_l[early_iter]),
		xytext=(early_iter, 0.5),
		arrowprops=dict(facecolor='black', shrink=0.05)
	)
	ax.set_xlabel('Iterations')
	ax.set_ylabel('Accuracy')
	ax.legend()
	plt.show()

def dot5_a():
	x_train_vec, y_train, y_train_oh, x_test_vec, y_test, y_test_oh = get_data(device='cuda')

	layers = [784, 100, 100, 10]
	niter = 2_000
	lr = 0.005
	reg = 0
	total = len(x_train_vec)
	batch_sizes = [int(0.001 * total), int(0.01 * total), int(0.05 * total), int(0.1 * total), int(0.2 * total), int(0.5 * total), total]
	shuffle = True
	print_debug = False

	acc_train_l = []
	acc_test_l = []
	losses_l = []

	for i, batch_size in enumerate(batch_sizes):
		print(f'{i=}, {batch_size=}')

		model = pt_deep.PTDeep(layers, torch.relu, device='cuda')
		losses = pt_deep.train_mb(model, x_train_vec, y_train_oh, batch_size, niter, lr, reg, shuffle, print_debug)
		losses_l.append(losses)

		train_probs = pt_deep.eval(model, x_train_vec)
		y_train_pred = np.argmax(train_probs, axis=1)
		acc_train, _, _, _ = data.eval_perf_multi(y_train.numpy(force=True), y_train_pred)
		test_probs = pt_deep.eval(model, x_test_vec)
		y_test_pred = np.argmax(test_probs, axis=1)
		acc_test, _, _, _ = data.eval_perf_multi(y_test.numpy(force=True), y_test_pred)

		acc_train_l.append(round(acc_train, 2))
		acc_test_l.append(round(acc_test, 2))

	fig = plt.figure(layout='constrained')
	fig.suptitle(f'mnist: {layers=}, {niter=}, {lr=}, {reg=}, {total=}, {shuffle=}')
	subfigs = fig.subfigures(nrows=2)
	loss_subfig = subfigs[0]
	axs = loss_subfig.subplots(ncols=len(batch_sizes))
	for i, batch_size in enumerate(batch_sizes):
		ax = axs[i]
		ax.plot(losses_l[i])
		ax.set_title(f'{batch_size=}\nloss={round(losses_l[i][-1], 3)}')
		ax.set_xlabel('Iterations')
		ax.set_ylabel('Loss')
	acc_subfig = subfigs[1]
	ax = acc_subfig.subplots()
	ax.set_title('Accuracy by batch sizes')
	labels = ['Train', 'Test']
	values = [acc_train_l, acc_test_l]
	width = 1
	ticks = np.linspace(0, (3 * width) * (len(batch_sizes) - 1), len(batch_sizes))
	ticks_names = [f'{batch_size}' for batch_size in batch_sizes]
	offset = 0
	for i in range(2):
		b = ax.bar(ticks + offset, values[i], width, label=labels[i])
		ax.bar_label(b)
		offset += width
	ax.set_ylabel('Accuracy')
	ax.set_xticks(ticks + width/2, ticks_names)
	ax.legend()
	plt.show()

def dot5_b():
	x_train_vec, y_train, y_train_oh, x_test_vec, y_test, y_test_oh = get_data()
	val_threshold = int(0.2 * len(x_train_vec))
	x_val_vec = x_train_vec[:val_threshold]
	x_nonval_vec = x_train_vec[val_threshold:]
	y_val = y_train[:val_threshold]
	y_nonval = y_train[val_threshold:]
	y_val_oh = y_train_oh[:val_threshold]
	y_nonval_oh = y_train_oh[val_threshold:]

	layers = [784, 100, 100, 10]
	niter = 20_000
	lr = 0.005
	reg = 0
	batch_size = int(0.05 * len(x_train_vec))
	shuffle = True
	print_debug = True

	model1 = pt_deep.PTDeep(layers, torch.relu, device='cuda')
	early_model = None
	early_acc = 0
	early_iter = 0
	def callback(iter: int):
		nonlocal model1, x_val_vec, y_val, early_model, early_acc, early_iter
		probs = pt_deep.eval(model1, x_val_vec)
		classes = np.argmax(probs, axis=1)
		acc, _, _, _ = data.eval_perf_multi(y_val.numpy(force=True), classes)
		if acc > early_acc:
			early_acc = acc
			early_model = copy.deepcopy(model1)
			early_iter = iter
	early_losses = pt_deep.train(model1, x_nonval_vec, y_nonval_oh, niter, lr, reg, print_debug, callback)

	train_probs = pt_deep.eval(early_model, x_train_vec)
	y_train_pred = np.argmax(train_probs, axis=1)
	acc_train_e, _, _, _ = data.eval_perf_multi(y_train.numpy(force=True), y_train_pred)
	test_probs = pt_deep.eval(early_model, x_test_vec)
	y_test_pred = np.argmax(test_probs, axis=1)
	acc_test_e, _, _, _ = data.eval_perf_multi(y_test.numpy(force=True), y_test_pred)
	print(f'Early model ({early_iter}/{niter}): Acc Train = {round(acc_train_e, 3)}, Acc Test = {round(acc_test_e, 3)}')

	model2 = pt_deep.PTDeep(layers, torch.relu, device='cuda')
	batched_losses = pt_deep.train_mb(model2, x_train_vec, y_train_oh, batch_size, niter, lr, reg, shuffle, print_debug, callback)

	train_probs = pt_deep.eval(model2, x_train_vec)
	y_train_pred = np.argmax(train_probs, axis=1)
	acc_train_b, _, _, _ = data.eval_perf_multi(y_train.numpy(force=True), y_train_pred)
	test_probs = pt_deep.eval(model2, x_test_vec)
	y_test_pred = np.argmax(test_probs, axis=1)
	acc_test_b, _, _, _ = data.eval_perf_multi(y_test.numpy(force=True), y_test_pred)
	print(f'Batched & shuffled model: Acc Train = {round(acc_train_b, 3)}, Acc Test = {round(acc_test_b, 3)}')

	fig = plt.figure(layout='constrained')
	fig.suptitle(f'minst: {layers=}, {niter=}, {lr=}\n{reg=}, {batch_size=}, total={len(x_train_vec)}, {shuffle=}')
	subfigs = fig.subfigures(nrows=2)
	loss_subfig = subfigs[0]
	loss_subfig.suptitle('Losses')
	axs = loss_subfig.subplots(ncols=2)
	axs[0].plot(early_losses)
	axs[0].set_title(f'Early Stopping Model\nLoss = {round(early_losses[early_iter], 3)}')
	axs[0].set_xlabel('Iterations')
	axs[0].set_ylabel('Loss')
	axs[1].plot(batched_losses)
	axs[1].set_title(f'Batched & Shuffled Model\nLoss = {round(batched_losses[-1], 3)}')
	axs[1].set_xlabel('Iterations')
	axs[1].set_ylabel('Loss')
	acc_subfig = subfigs[1]
	acc_subfig.suptitle('Accuracy')
	ax = acc_subfig.subplots()
	values = [[acc_train_e, acc_test_e], [acc_train_b, acc_test_b]]
	labels = ['Early', 'Batched & Shuffled']
	width = 1
	ticks = np.linspace(0, (len(labels) + 1) * width, 2)
	offset = 0
	for i in range(2):
		b = ax.bar(ticks + offset, values[i], width, label=labels[i])
		ax.bar_label(b)
		offset += width
	ax.set_ylabel('Accuracy')
	ax.set_xticks(ticks + width/2, ['Train', 'Test'])
	ax.legend()
	plt.show()

def dot6():
	x_train_vec, y_train, y_train_oh, x_test_vec, y_test, y_test_oh = get_data(device='cuda')

	layers = [784, 100, 100, 10]
	niter = 1_000
	lrs = [1e-3, 1e-4, 1e-5]
	reg = 0
	total = len(x_train_vec)
	batch_size = int(0.01 * total)
	shuffle = True
	print_debug = False

	acc_l = []
	acc_labels = []
	losses_l = []
	losses_labels = []

	for i, lr in enumerate(lrs):
		print(f'{i=}, Adam, {lr=}')

		model = pt_deep.PTDeep(layers, torch.relu, device='cuda')
		losses = pt_deep.train_mb_adam(model, x_train_vec, y_train_oh, batch_size, niter, lr, reg, shuffle, print_debug)
		losses_l.append(losses)
		losses_labels.append(f'Adam, {lr=}, Loss={round(losses[-1], 3)}')

		train_probs = pt_deep.eval(model, x_train_vec)
		y_train_pred = np.argmax(train_probs, axis=1)
		acc_train, _, _, _ = data.eval_perf_multi(y_train.numpy(force=True), y_train_pred)
		test_probs = pt_deep.eval(model, x_test_vec)
		y_test_pred = np.argmax(test_probs, axis=1)
		acc_test, _, _, _ = data.eval_perf_multi(y_test.numpy(force=True), y_test_pred)

		acc_l.append([round(acc_train, 3), round(acc_test, 3)])
		acc_labels.append(f'Adam, {lr=}')
	for i, lr in enumerate(lrs):
		print(f'{i=}, SGD, {lr=}')

		model = pt_deep.PTDeep(layers, torch.relu, device='cuda')
		losses = pt_deep.train_mb(model, x_train_vec, y_train_oh, batch_size, niter, lr, reg, shuffle, print_debug)
		losses_l.append(losses)
		losses_labels.append(f'SGD, {lr=}, Loss={round(losses[-1], 3)}')

		train_probs = pt_deep.eval(model, x_train_vec)
		y_train_pred = np.argmax(train_probs, axis=1)
		acc_train, _, _, _ = data.eval_perf_multi(y_train.numpy(force=True), y_train_pred)
		test_probs = pt_deep.eval(model, x_test_vec)
		y_test_pred = np.argmax(test_probs, axis=1)
		acc_test, _, _, _ = data.eval_perf_multi(y_test.numpy(force=True), y_test_pred)

		acc_l.append([round(acc_train, 3), round(acc_test, 3)])
		acc_labels.append(f'SGD, {lr=}')

	fig = plt.figure(layout='constrained')
	fig.suptitle(f'mnist: {layers=}, {niter=}, {reg=}, {batch_size=}, {total=}, {shuffle=}')
	(loss_ax, acc_ax) = fig.subplots(nrows=2)
	loss_ax.set_title('Losses for different algorithms and learning rates')
	loss_ax.set_xlabel('Iterations')
	loss_ax.set_ylabel('Loss')
	for i, losses in enumerate(losses_l):
		loss_ax.plot(losses, label=losses_labels[i])
	loss_ax.legend()
	width = 1
	ticks = np.linspace(0, (2*len(lrs)+1) * width, 2)
	ticks_names = ['Train', 'Test']
	offset = 0
	for i in range(2*len(lrs)):
		b = acc_ax.bar(ticks + offset, acc_l[i], width, label=acc_labels[i])
		acc_ax.bar_label(b)
		offset += width
	acc_ax.set_ylabel('Accuracy')
	acc_ax.set_xticks(ticks + len(lrs) * width - width/2, ticks_names)
	acc_ax.legend()
	plt.show()

def dot7():
	x_train_vec, y_train, y_train_oh, x_test_vec, y_test, y_test_oh = get_data(device='cuda')

	layers = [784, 100, 100, 10]
	niter = 250
	lrs = [1e-3, 1e-4, 1e-5]
	reg = 0
	total = len(x_train_vec)
	batch_size = int(0.01 * total)
	shuffle = True
	print_debug = False
	gamma = 1-1e-6

	acc_l = []
	acc_labels = []
	losses_l = []
	losses_labels = []

	for i, lr in enumerate(lrs):
		print(f'{i=}, Adam, {lr=}')

		model = pt_deep.PTDeep(layers, torch.relu, device='cuda')
		losses = pt_deep.train_mb_adam(model, x_train_vec, y_train_oh, batch_size, niter, lr, reg, shuffle, print_debug)
		losses_l.append(losses)
		losses_labels.append(f'Adam, {lr=}, Loss={round(losses[-1], 3)}')

		train_probs = pt_deep.eval(model, x_train_vec)
		y_train_pred = np.argmax(train_probs, axis=1)
		acc_train, _, _, _ = data.eval_perf_multi(y_train.numpy(force=True), y_train_pred)
		test_probs = pt_deep.eval(model, x_test_vec)
		y_test_pred = np.argmax(test_probs, axis=1)
		acc_test, _, _, _ = data.eval_perf_multi(y_test.numpy(force=True), y_test_pred)

		acc_l.append([round(acc_train, 3), round(acc_test, 3)])
		acc_labels.append(f'Adam, {lr=}')
	for i, lr in enumerate(lrs):
		print(f'{i=}, Adam Adapt, {lr=}')

		model = pt_deep.PTDeep(layers, torch.relu, device='cuda')
		losses = pt_deep.train_mb_adam_adapt(model, x_train_vec, y_train_oh, batch_size, niter, lr, reg, shuffle, print_debug, gamma)
		losses_l.append(losses)
		losses_labels.append(f'Adam Adapt, {lr=}, Loss={round(losses[-1], 3)}')

		train_probs = pt_deep.eval(model, x_train_vec)
		y_train_pred = np.argmax(train_probs, axis=1)
		acc_train, _, _, _ = data.eval_perf_multi(y_train.numpy(force=True), y_train_pred)
		test_probs = pt_deep.eval(model, x_test_vec)
		y_test_pred = np.argmax(test_probs, axis=1)
		acc_test, _, _, _ = data.eval_perf_multi(y_test.numpy(force=True), y_test_pred)

		acc_l.append([round(acc_train, 3), round(acc_test, 3)])
		acc_labels.append(f'Adam Adapt, {lr=}')

	fig = plt.figure(layout='constrained')
	fig.suptitle(f'mnist: {layers=}, {niter=}, {reg=}, {batch_size=}, {total=}, {shuffle=}, {gamma=}')
	(loss_ax, acc_ax) = fig.subplots(nrows=2)
	loss_ax.set_title('Losses for different algorithms and (initial) learning rates')
	loss_ax.set_xlabel('Iterations')
	loss_ax.set_ylabel('Loss')
	for i, losses in enumerate(losses_l):
		loss_ax.plot(losses, label=losses_labels[i])
	loss_ax.legend()
	width = 1
	ticks = np.linspace(0, (2*len(lrs)+1) * width, 2)
	ticks_names = ['Train', 'Test']
	offset = 0
	for i in range(2*len(lrs)):
		b = acc_ax.bar(ticks + offset, acc_l[i], width, label=acc_labels[i])
		acc_ax.bar_label(b)
		offset += width
	acc_ax.set_ylabel('Accuracy')
	acc_ax.set_xticks(ticks + len(lrs) * width - width/2, ticks_names)
	acc_ax.legend()
	plt.show()

def dot9():
	x_train_vec, y_train, y_train_oh, x_test_vec, y_test, y_test_oh = get_data(device='cpu')

	percent = 100
	total = len(x_train_vec)
	threshold = int(percent / 100 * total)

	print(f'Dataset: {percent}% ({threshold}/{total})')

	rbf_svm = svm.SVC()
	rbf_svm.fit(x_train_vec[:threshold], y_train[:threshold])
	y_train_pred = rbf_svm.predict(x_train_vec)
	rbf_acc_train, _, _, _ = data.eval_perf_multi(y_train.numpy(force=True), y_train_pred)
	y_test_pred = rbf_svm.predict(x_test_vec)
	rbf_acc_test, _, _, _ = data.eval_perf_multi(y_test.numpy(force=True), y_test_pred)

	print(f'RBF SVM: Accuracy: Train={round(rbf_acc_train, 3)}, Test={round(rbf_acc_test, 3)}')

	linear_svm = svm.SVC(kernel='linear')
	linear_svm.fit(x_train_vec, y_train)
	y_train_pred = linear_svm.predict(x_train_vec)
	linear_acc_train, _, _, _ = data.eval_perf_multi(y_train.numpy(force=True), y_train_pred)
	y_test_pred = linear_svm.predict(x_test_vec)
	linear_acc_test, _, _, _ = data.eval_perf_multi(y_test.numpy(force=True), y_test_pred)

	print(f'Linear SVM: Accuracy: Train={round(linear_acc_train, 3)}, Test={round(linear_acc_test, 3)}')

	x_train_vec, y_train, y_train_oh, x_test_vec, y_test, y_test_oh = get_data()
	layers = [784, 100, 100, 10]
	niter = 4_000
	lr = 1e-3
	reg = 1e-3
	batch_size = int(0.01 * total)
	shuffle = True
	print_debug = False
	deep = pt_deep.PTDeep(layers, torch.relu, device='cuda')
	pt_deep.train_mb_adam(deep, x_train_vec[:threshold], y_train_oh[:threshold], batch_size, niter, lr, reg, shuffle, print_debug)

	train_probs = pt_deep.eval(deep, x_train_vec)
	y_train_pred = np.argmax(train_probs, axis=1)
	deep_acc_train, _, _, _ = data.eval_perf_multi(y_train.numpy(force=True), y_train_pred)
	test_probs = pt_deep.eval(deep, x_test_vec)
	y_test_pred = np.argmax(test_probs, axis=1)
	deep_acc_test, _, _, _ = data.eval_perf_multi(y_test.numpy(force=True), y_test_pred)

	print(f'Deep ANN: Accuracy: Train={round(deep_acc_train, 3)}, Test={round(deep_acc_test, 3)}')

	fig = plt.figure(layout='constrained')
	fig.suptitle("RBF SVM vs. Linear SVM vs. Deep ANN on MNIST dataset")
	ax = fig.subplots()
	width = 1
	ticks = np.linspace(0, 4 * width, 2)
	ticks_labels = ["Train", "Test"]
	offset = 0
	values = [
		(round(rbf_acc_train, 3),round(rbf_acc_test, 3)),
		(round(linear_acc_train, 3), round(linear_acc_test, 3)),
		(round(deep_acc_train, 3), round(deep_acc_test, 3))
	]
	labels = ["RBF SVM", "Linear SVM", "Deep ANN"]
	for i, value in enumerate(values):
		b = ax.bar(ticks + offset, value, width=width, label=labels[i])
		ax.bar_label(b)
		offset += width
	ax.set_xticks(ticks + width, ticks_labels)
	ax.legend()
	ax.set_ylabel('Accuracy')
	plt.show()

if __name__ == '__main__':
	dot9()
