import torch
import numpy as np

from typing import Tuple

class AvgPoolAll(torch.nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		return input.mean(dim=1)

class Baseline(torch.nn.Module):
	def __init__(self, embedding: torch.nn.Module):
		super().__init__()
		self.stack = torch.nn.Sequential(
			embedding,
			AvgPoolAll(),
			torch.nn.Linear(300, 150),
			torch.nn.ReLU(),
			torch.nn.Linear(150, 150),
			torch.nn.ReLU(),
			torch.nn.Linear(150, 1)
		)
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		return self.stack(input)

def load_dataset(
		batch_size: int = 32,
		shuffle: bool = True,
		max_vocab_size: int = -1,
		min_vocab_freq: int = 0,
		device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	from nlp_dataset import NLPDataset, pad_collate_fn

	train_dataset = NLPDataset('data/sst_train_raw.csv')
	valid_dataset = NLPDataset('data/sst_valid_raw.csv')
	test_dataset = NLPDataset('data/sst_test_raw.csv')

	from data import get_frequencies_text, get_embedding_matrix
	freq = get_frequencies_text(train_dataset.instances)

	from vocab import Vocab
	vocab = Vocab(freq, max_vocab_size, min_vocab_freq)

	train_dataset.vocab = vocab
	valid_dataset.vocab = vocab
	test_dataset.vocab = vocab

	train_dataloader = torch.utils.data.DataLoader(
		dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate_fn)
	valid_dataloader = torch.utils.data.DataLoader(
		dataset=valid_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate_fn)
	test_dataloader = torch.utils.data.DataLoader(
		dataset=test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate_fn)
	
	embedding = get_embedding_matrix(vocab, 'data/sst_glove_6b_300d.txt', device)

	return train_dataloader, valid_dataloader, test_dataloader, embedding

def train(model: torch.nn.Module, data: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion):
	model.train()

	max_loss = 0
	for batch_num, batch in enumerate(data):
		model.zero_grad()

		texts, labels, lengths = batch

		logits = model(texts)
		loss = criterion(logits, labels)
		loss.backward()
		optimizer.step()

		max_loss = max(max_loss, loss.item())
	return max_loss

def evaluate(model: torch.nn.Module, data: torch.utils.data.DataLoader, criterion):
	model.eval()
	max_loss = 0
	predict = torch.tensor([], dtype=torch.int)
	correct = torch.tensor([], dtype=torch.int)
	with torch.no_grad():
		for batch_num, batch in enumerate(data):
			texts, labels, lengths = batch
			logits = model(texts)
			loss = criterion(logits, labels)
			max_loss = max(max_loss, loss.item())

			batch_predict = torch.round(torch.sigmoid(logits))
			batch_correct = labels
			predict = torch.cat([predict, batch_predict.to(dtype=torch.int)])
			correct = torch.cat([correct, batch_correct.to(dtype=torch.int)])
	import measures
	accuracy, precision, recall, f1, confusion = measures.eval_perf_binary(predict.numpy(force=True), correct.numpy(force=True))
	return max_loss, accuracy, precision, recall, f1, confusion

if __name__ == '__main__':
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
	args = parser.parse_args()

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	train_ds, valid_ds, test_ds, embedding = load_dataset(
		args.batch_size, args.shuffle, args.max_vocab_size, args.min_vocab_freq, args.device)
	
	model = Baseline(embedding)
	model.to(device=args.device)
	
	criterion = torch.nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	for epoch in range(args.epochs):
		train(model, train_ds, optimizer, criterion)
		loss, accuracy, precision, recall, f1, confusion = evaluate(model, valid_ds, criterion)
		print(f'{epoch=}, valid loss={round(loss, 3)}, acc={round(accuracy, 3)}, prec={round(precision, 3)}, rec={round(recall, 3)}, f1={round(f1, 3)}, {confusion=}')
	
	loss, accuracy, precision, recall, f1, confusion = evaluate(model, test_ds, criterion)
	# print(f'test loss={round(loss, 3)}, acc={round(accuracy, 3)}, prec={round(precision, 3)}, rec={round(recall, 3)}, , f1={round(f1, 3)}, {confusion=}')
	print(f'{args.seed} | {round(loss, 3)} | {round(accuracy, 3)} | {round(precision, 3)} | {round(recall, 3)} | {round(f1, 3)}')
