from typing import Type

import torch
import numpy as np

from baseline import load_dataset
import measures

class MyRNN(torch.nn.Module):
	def __init__(
			self,
			base: Type[torch.nn.RNN] | Type[torch.nn.GRU] | Type[torch.nn.LSTM],
			hidden_size: int = 150,
			num_layers: int = 2
		) -> None:
		super().__init__()
		self.base = base
		self.RNN = base(input_size=300, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
		self.FC = torch.nn.Sequential(
			torch.nn.Linear(hidden_size, 150),
			torch.nn.ReLU(),
			torch.nn.Linear(150, 1)
		)
	def forward(self, input: torch.nn.utils.rnn.PackedSequence) -> torch.Tensor:
		if self.base == torch.nn.LSTM:
			_, rest = self.RNN(input)
			h_t = rest[0]
		else:
			_, h_t = self.RNN(input)
		rnn_output: torch.Tensor = h_t[-1]
		out = self.FC(rnn_output)
		return out

def train(
	model: MyRNN,
	dataloader: torch.utils.data.DataLoader,
	embedding: torch.nn.Embedding,
	criterion,
	clip_norm: float,
	optimizer: torch.optim.Optimizer,
	device: torch.device
) -> float:
	model.train()
	max_loss: float = 0.0
	for batch in dataloader:
		optimizer.zero_grad()
		
		texts, labels, lengths = batch

		input = embedding(texts).to(device=device)
		labels = labels.to(device=device)
		packed_seq = torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True, enforce_sorted=False)
		
		logits = model(packed_seq)
		loss: torch.Tensor = criterion(logits, labels)
		loss.backward()
	
		torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
		optimizer.step()

		max_loss = max(max_loss, loss.item())

	return max_loss

def evaluate(
	model: MyRNN,
	dataloader: torch.utils.data.DataLoader,
	embedding: torch.nn.Embedding,
	criterion,
	device: torch.device,
) -> tuple[float, float, float, float, float, np.ndarray]:
	model.eval()
	
	with torch.no_grad():
		max_loss: float = 0.0
		correct = torch.tensor([], dtype=torch.int, device=device)
		predict = torch.tensor([], dtype=torch.int, device=device)
		for batch in dataloader:
			texts, labels, lengths = batch
			input = embedding(texts).to(device=device)
			packed_seq = torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True, enforce_sorted=False)
			labels = labels.to(device=device)

			logits = model(packed_seq)
			
			loss = criterion(logits, labels)
			max_loss = max(max_loss, loss.item())

			predicted = torch.round(torch.sigmoid(logits)).to(dtype=int)

			correct = torch.cat([correct, labels.to(dtype=int)])
			predict = torch.cat([predict, predicted.to(dtype=int)])
		accuracy, precision, recall, f1, confusion = measures.eval_perf_binary(predict.numpy(force=True), correct.numpy(force=True))
		return max_loss, accuracy, precision, recall, f1, confusion

def test():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", type=int, default=7052020)
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--shuffle", type=bool, default=True)
	parser.add_argument("--max_vocab_size", type=int, default=-1)
	parser.add_argument("--min_vocab_freq", type=int, default=0)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--device", type=str, default='cpu')
	parser.add_argument("--clip_norm", type=float, default=1.0)
	parser.add_argument("--hidden_size", type=int, default=150)
	parser.add_argument("--num_layers", type=int, default=2)
	args = parser.parse_args()

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	train_ds, valid_ds, test_ds, embedding = load_dataset(
		args.batch_size, args.shuffle, args.max_vocab_size, args.min_vocab_freq)
	
	model = MyRNN(args.hidden_size, args.num_layers).to(device=args.device)
	
	criterion = torch.nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	for epoch in range(args.epochs):
		train_loss = train(model, train_ds, embedding, criterion, args.clip_norm, optimizer, args.device)
		loss, accuracy, precision, recall, f1, confusion = evaluate(model, valid_ds, embedding, criterion, args.device)
		print(f'{epoch=:3}, train loss={train_loss:7.3f}, valid loss={loss:7.3f}, acc={accuracy:7.3f}, prec={precision:7.3f}, rec={recall:7.3f}, f1={f1:7.3f}')
	
	loss, accuracy, precision, recall, f1, confusion = evaluate(model, test_ds, embedding, criterion, args.device)
	print(f'test loss={loss:7.3f}, acc={accuracy:7.3f}, prec={precision:7.3f}, rec={recall:7.3f}, f1={f1:7.3f}')
	# print(f'{args.seed} | {loss:7.3f} | {accuracy:7.3f} | {precision:7.3f} | {recall:7.3f} | {f1:7.3f}')

def report1():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", type=int, default=7052020)
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--shuffle", type=bool, default=True)
	parser.add_argument("--max_vocab_size", type=int, default=-1)
	parser.add_argument("--min_vocab_freq", type=int, default=0)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--device", type=str, default='cpu')
	parser.add_argument("--clip_norm", type=float, default=1.0)
	parser.add_argument("--hidden_size", type=int, default=150)
	parser.add_argument("--num_layers", type=int, default=2)
	args = parser.parse_args()

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	train_ds, valid_ds, test_ds, embedding = load_dataset(args.batch_size, args.shuffle, args.max_vocab_size, args.min_vocab_freq)
	
	bases = {"RNN": torch.nn.RNN, "GRU": torch.nn.GRU, "LSTM": torch.nn.LSTM}
	train_losses = dict()
	train_accs = dict()
	test_loss = dict()
	test_acc = dict()
	for base_name, base_class in bases.items():
		print(base_name)

		model = MyRNN(base_class, args.hidden_size, args.num_layers).to(device=args.device)
		
		criterion = torch.nn.BCEWithLogitsLoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

		train_loss = []
		train_acc = []
		for epoch in range(args.epochs):
			print(f'\t{epoch + 1}/{args.epochs}')
			train(model, train_ds, embedding, criterion, args.clip_norm, optimizer, args.device)
			loss, acc, _, _, _, _ = evaluate(model, train_ds, embedding, criterion, args.device)
			train_loss.append(loss)
			train_acc.append(acc)
		train_losses[base_name] = train_loss
		train_accs[base_name] = train_acc
		
		loss, acc, _, _, _, _ = evaluate(model, test_ds, embedding, criterion, args.device)
		test_loss[base_name] = loss
		test_acc[base_name] = acc

		del optimizer, criterion, model
	
	import matplotlib.pyplot as plt
	fig = plt.figure(layout='constrained')
	fig.suptitle(
		f'seed={args.seed}, batch_size={args.batch_size}, shuffle={args.shuffle}\n' + 
	    f'vocab_size={args.max_vocab_size}, vocab_freq={args.min_vocab_freq}\n' +
		f'lr={args.lr}, grad_clip={args.clip_norm}\n' +
		f'hidden_size={args.hidden_size}, num_layers={args.num_layers}'
	)
	axs = fig.subplots(nrows=2)
	loss_ax: plt.Axes = axs[0]
	loss_ax.set_title('Train losses')
	loss_ax.set_xlabel('Epoch')
	loss_ax.set_ylabel('Loss')
	for base_name in bases.keys():
		losses = train_losses[base_name]
		loss_ax.plot(losses, label=f'{base_name}: {losses[-1]:7.3f}')
	loss_ax.legend()
	acc_ax: plt.Axes = axs[1]
	acc_ax.set_title('Accuracies')
	acc_ax.set_ylabel('Accuracy')
	xticks = np.array([0, 4])
	offset = 0
	for base_name in bases.keys():
		accs = [round(train_accs[base_name][-1], 3), round(test_acc[base_name], 3)]
		b = acc_ax.bar(xticks + offset, accs, label=base_name, width=1)
		acc_ax.bar_label(b)
		offset += 1
	xlabels = ['Train', 'Test']
	acc_ax.set_xticks(xticks + 1, xlabels)
	acc_ax.legend()
	plt.show()

def report2():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", type=int, default=7052020)
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--shuffle", type=bool, default=True)
	parser.add_argument("--max_vocab_size", type=int, default=-1)
	parser.add_argument("--min_vocab_freq", type=int, default=0)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--device", type=str, default='cpu')
	parser.add_argument("--clip_norm", type=float, default=1.0)
	parser.add_argument("--hidden_size", type=int, default=150)
	parser.add_argument("--num_layers", type=int, default=2)
	args = parser.parse_args()

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	train_ds, valid_ds, test_ds, embedding = load_dataset(args.batch_size, args.shuffle, args.max_vocab_size, args.min_vocab_freq)
	
	bases = {"RNN": torch.nn.RNN, "GRU": torch.nn.GRU, "LSTM": torch.nn.LSTM}
	hidden_sizes = [90, 120, 150, 180, 210]

	train_accs = dict()
	test_accs = dict()
	for base_name, base_class in bases.items():
		train_acc = []
		test_acc = []
		for hidden_size in hidden_sizes:
			print(f'{base_name=}, {hidden_size=}')

			model = MyRNN(base_class, hidden_size, args.num_layers).to(device=args.device)
			
			criterion = torch.nn.BCEWithLogitsLoss()
			optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

			for epoch in range(args.epochs):
				print(f'\t{epoch + 1}/{args.epochs}')
				train(model, train_ds, embedding, criterion, args.clip_norm, optimizer, args.device)
			
			_, tracc, _, _, _, _ = evaluate(model, train_ds, embedding, criterion, args.device)
			_, teacc, _, _, _, _ = evaluate(model, test_ds, embedding, criterion, args.device)
			train_acc.append(round(tracc, 3))
			test_acc.append(round(teacc, 3))

			del optimizer, criterion, model
		train_accs[base_name] = train_acc
		test_accs[base_name] = test_acc
	
	import matplotlib.pyplot as plt
	fig = plt.figure(layout='constrained')
	fig.suptitle(
		f'seed={args.seed}, batch_size={args.batch_size}, shuffle={args.shuffle} ' + 
	    f'vocab_size={args.max_vocab_size}, vocab_freq={args.min_vocab_freq} ' +
		f'lr={args.lr}, grad_clip={args.clip_norm} ' +
		f'num_layers={args.num_layers}'
	)
	axs = fig.subplots(nrows=3)
	for i, base_name in enumerate(bases.keys()):
		ax: plt.Axes = axs[i]
		ax.set_title(base_name)
		ax.set_ylabel('Accuracy')
		xticks = np.arange(len(hidden_sizes)) * 3
		offset = 0
		for label, values in zip(['Train', 'Test'], [train_accs[base_name], test_accs[base_name]]):
			b = ax.bar(xticks + offset, values, width=1, label=label)
			ax.bar_label(b)
			offset += 1
		ax.set_xticks(xticks + 0.5, [str(h) for h in hidden_sizes])
		ax.legend()
		
	plt.show()

def report3():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", type=int, default=7052020)
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--shuffle", type=bool, default=True)
	parser.add_argument("--max_vocab_size", type=int, default=-1)
	parser.add_argument("--min_vocab_freq", type=int, default=0)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--device", type=str, default='cpu')
	parser.add_argument("--clip_norm", type=float, default=1.0)
	parser.add_argument("--hidden_size", type=int, default=150)
	parser.add_argument("--num_layers", type=int, default=2)
	args = parser.parse_args()

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	train_ds, valid_ds, test_ds, embedding = load_dataset(args.batch_size, args.shuffle, args.max_vocab_size, args.min_vocab_freq)
	
	bases = {"RNN": torch.nn.RNN, "GRU": torch.nn.GRU, "LSTM": torch.nn.LSTM}
	num_layers_l = [1, 2, 4, 8, 16]

	train_accs = dict()
	test_accs = dict()
	for base_name, base_class in bases.items():
		train_acc = []
		test_acc = []
		for num_layers in num_layers_l:
			print(f'{base_name=}, {num_layers=}')

			model = MyRNN(base_class, args.hidden_size, num_layers).to(device=args.device)
			
			criterion = torch.nn.BCEWithLogitsLoss()
			optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

			for epoch in range(args.epochs):
				print(f'\t{epoch + 1}/{args.epochs}')
				train(model, train_ds, embedding, criterion, args.clip_norm, optimizer, args.device)
			
			_, tracc, _, _, _, _ = evaluate(model, train_ds, embedding, criterion, args.device)
			_, teacc, _, _, _, _ = evaluate(model, test_ds, embedding, criterion, args.device)
			train_acc.append(round(tracc, 3))
			test_acc.append(round(teacc, 3))

			del optimizer, criterion, model
		train_accs[base_name] = train_acc
		test_accs[base_name] = test_acc
	
	import matplotlib.pyplot as plt
	fig = plt.figure(layout='constrained')
	fig.suptitle(
		f'seed={args.seed}, batch_size={args.batch_size}, shuffle={args.shuffle} ' + 
	    f'vocab_size={args.max_vocab_size}, vocab_freq={args.min_vocab_freq} ' +
		f'lr={args.lr}, grad_clip={args.clip_norm} ' +
		f'hidden_size={args.hidden_size}'
	)
	axs = fig.subplots(nrows=3)
	for i, base_name in enumerate(bases.keys()):
		ax: plt.Axes = axs[i]
		ax.set_title(base_name)
		ax.set_ylabel('Accuracy')
		xticks = np.arange(len(num_layers_l)) * 3
		offset = 0
		for label, values in zip(['Train', 'Test'], [train_accs[base_name], test_accs[base_name]]):
			b = ax.bar(xticks + offset, values, width=1, label=label)
			ax.bar_label(b)
			offset += 1
		ax.set_xticks(xticks + 0.5, [str(n) for n in num_layers_l])
		ax.legend()
		
	plt.show()

if __name__ == "__main__":
	report3()
