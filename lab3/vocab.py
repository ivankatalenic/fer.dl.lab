import torch

class Vocab:
	"""
	Creates a vocabulary from the word frequency dictionary.

	Has methods for converting words to indices within a vocabulary.
	"""
	def __init__(self, freq: dict[str, int], max_size: int = -1, min_freq: int = 0):
		self.stoi: dict[str, int] = {'<PAD>': 0, '<UNK>': 1}
		words_sorted = sorted(list(freq), key=lambda w: freq[w], reverse=True)
		next_index = 2
		for w in words_sorted:
			if max_size != -1 and len(self.stoi) >= max_size or freq[w] < min_freq:
				break
			self.stoi[w] = next_index
			next_index += 1
	def encode(self, words: list[str]) -> torch.Tensor:
		"""Returns an array of indices for words in the input list."""
		ret = torch.empty((len(words), ), dtype=torch.int)
		for i, word in enumerate(words):
			if word not in self.stoi:
				ret[i] = self.stoi['<UNK>']
				continue
			ret[i] = self.stoi[word]
		return ret

if __name__ == '__main__':
	from nlp_dataset import *
	from data import *

	ds = NLPDataset('data/sst_train_raw.csv')
	freq = get_frequencies_text(ds.instances)
	vocab = Vocab(freq)
	print(f'{len(vocab.stoi)=}')
	words = ['the', 'a', 'and', 'my', 'twists', 'lets', 'sports', 'amateurishly']
	print(f'{vocab.encode(words)}')
