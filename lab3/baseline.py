import torch
import numpy as np

class AvgPoolAll(torch.nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, args: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
		input, lengths = args
		return input.sum(dim=1).div_(lengths.reshape(-1, 1))

class Baseline(torch.nn.Module):
	def __init__(self, embedding: torch.nn.Module):
		super().__init__()
		self.embedding = embedding
		self.stack = torch.nn.Sequential(
			AvgPoolAll(),
			torch.nn.Linear(300, 150),
			torch.nn.ReLU(),
			torch.nn.Linear(150, 150),
			torch.nn.ReLU(),
			torch.nn.Linear(150, 1)
		)
	def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
		embed = self.embedding(input)
		return self.stack((embed, lengths))

def load_dataset(
		batch_size: int = 32,
		shuffle: bool = True,
		max_vocab_size: int = -1,
		min_vocab_freq: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.nn.Embedding]:
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
	
	embedding = get_embedding_matrix(vocab, 'data/sst_glove_6b_300d.txt')

	return train_dataloader, valid_dataloader, test_dataloader, embedding

def train(
	model: torch.nn.Module,
	data: torch.utils.data.DataLoader,
	optimizer: torch.optim.Optimizer,
	criterion,
	device: torch.device = 'cpu'
)-> float:
	model.train()

	max_loss = torch.tensor(0.0, device=device)
	for batch_num, batch in enumerate(data):
		model.zero_grad()

		texts, labels, lengths = batch
		texts = texts.to(device=device)
		labels = labels.to(device=device)
		lengths = lengths.to(device=device)

		logits = model(texts, lengths)
		loss = criterion(logits, labels)
		loss.backward()
		optimizer.step()

		max_loss = torch.max(max_loss, loss)
	return max_loss.item()

def evaluate(
	model: torch.nn.Module,
	data: torch.utils.data.DataLoader,
	criterion,
	device: torch.device = 'cpu'
) -> tuple[float, float, float, float, float, np.ndarray]:
	model.eval()

	max_loss = torch.tensor(0.0, device=device)
	predict = torch.tensor([], dtype=torch.int, device=device)
	correct = torch.tensor([], dtype=torch.int, device=device)
	with torch.no_grad():
		for batch_num, batch in enumerate(data):
			texts, labels, lengths = batch
			texts = texts.to(device=device)
			labels = labels.to(device=device)
			lengths = lengths.to(device=device)

			logits = model(texts, lengths)
			loss = criterion(logits, labels)
			max_loss = torch.max(max_loss, loss)

			batch_predict = torch.round(torch.sigmoid(logits))
			batch_correct = labels
			predict = torch.cat([predict, batch_predict.to(dtype=torch.int)])
			correct = torch.cat([correct, batch_correct.to(dtype=torch.int)])
	import measures
	accuracy, precision, recall, f1, confusion = measures.eval_perf_binary(predict.numpy(force=True), correct.numpy(force=True))
	return max_loss.item(), accuracy, precision, recall, f1, confusion

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
		args.batch_size, args.shuffle, args.max_vocab_size, args.min_vocab_freq)
	embedding = embedding.to(device=args.device)
	
	model = Baseline(embedding).to(device=args.device)
	
	criterion = torch.nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	for epoch in range(args.epochs):
		train_loss = train(model, train_ds, optimizer, criterion, args.device)
		loss, accuracy, precision, recall, f1, confusion = evaluate(model, valid_ds, criterion, args.device)
		print(f'{epoch=:3}, train loss={train_loss:7.3f}, valid loss={loss:7.3f}, acc={accuracy:7.3f}, prec={precision:7.3f}, rec={recall:7.3f}, f1={f1:7.3f}')
	
	loss, accuracy, precision, recall, f1, confusion = evaluate(model, test_ds, criterion, args.device)
	print(f'test loss={loss:7.3f}, acc={accuracy:7.3f}, prec={precision:7.3f}, rec={recall:7.3f}, f1={f1:7.3f}')
	# print(f'{args.seed} | {loss:7.3f} | {accuracy:7.3f} | {precision:7.3f} | {recall:7.3f} | {f1:7.3f}')
