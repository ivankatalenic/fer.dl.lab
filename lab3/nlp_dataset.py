import torch

from data_instance import DataInstance
from vocab import Vocab

class NLPDataset(torch.utils.data.Dataset):
	def __init__(self, filepath: str, vocab: Vocab = None):
		self.load_instances(filepath)
		self.vocab: Vocab = vocab
	def load_instances(self, filepath: str):
		self.instances: list[DataInstance] = []
		with open(filepath) as f:
			for line in f:
				tokens: list[str] = line.strip().split(', ')
				self.instances.append(DataInstance(tokens[0].split(' '), tokens[1]))
	def __getitem__(self, key: int):
		instance = self.instances[key]
		label_idx = 0 if instance.label == 'positive' else 1
		return self.vocab.encode(instance.text), torch.tensor(label_idx, dtype=torch.int)
	def __len__(self):
		return len(self.instances)
	
def test1():
	from data import get_frequencies_text
	ds = NLPDataset('data/sst_test_raw.csv')
	freq = get_frequencies_text(ds.instances)
	ds.vocab = Vocab(freq)
	print(ds.instances[871])
	print(ds[871])

if __name__ == "__main__":
	from data import get_frequencies_text, pad_collate_fn

	batch_size = 2 # Only for demonstrative purposes
	shuffle = False # Only for demonstrative purposes
	train_dataset = NLPDataset('data/sst_train_raw.csv')
	freq = get_frequencies_text(train_dataset.instances)
	train_dataset.vocab = Vocab(freq)
	train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, 
								shuffle=shuffle, collate_fn=pad_collate_fn)
	texts, labels, lengths = next(iter(train_dataloader))
	print(f"Texts: {texts}")
	print(f"Labels: {labels}")
	print(f"Lengths: {lengths}")
