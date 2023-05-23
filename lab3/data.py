from vocab import Vocab
import torch

def get_frequencies_text(instances):
	freq = {}
	for instance in instances:
		words = instance.text
		for w in words:
			if w not in freq:
				freq[w] = 0
			freq[w] += 1
	return freq

def _freq_test():
	from nlp_dataset import NLPDataset
	
	ds = NLPDataset('data/sst_train_raw.csv')
	freq = get_frequencies_text(ds)
	print(f'{len(freq)=}')
	words_sorted = sorted(list(freq), key=lambda w: freq[w], reverse=True)
	top10 = {w: freq[w] for w in words_sorted[:10]}
	print(f'{top10=}')

def get_embedding_matrix(vocab: Vocab, repr_file: str) -> torch.nn.Embedding:
	word_repr = {}
	with open(repr_file, buffering=2**30) as f:
		for line in f:
			line = line.strip()
			tokens = line.split(' ')
			word = tokens[0]
			vector = torch.empty((300,), dtype=torch.float)
			for i, elem in enumerate(tokens[1:]):
				vector[i] = float(elem)
			word_repr[word] = vector
	embed = torch.randn((len(vocab.stoi), 300), dtype=torch.float)
	for word, index in vocab.stoi.items():
		if word not in word_repr:
			continue
		vector = word_repr[word]
		embed[index] = vector
	embed[0] = torch.zeros((300,), dtype=torch.float)
	return torch.nn.Embedding.from_pretrained(embed)

if __name__ == '__main__':
	from nlp_dataset import NLPDataset
	from vocab import Vocab
	
	ds = NLPDataset('data/sst_train_raw.csv')
	freq = get_frequencies_text(ds)
	print(f'{len(freq)=}')
	words_sorted = sorted(list(freq), key=lambda w: freq[w], reverse=True)
	top10 = {w: freq[w] for w in words_sorted[:10]}
	print(f'{top10=}')

	vocab = Vocab(freq, 50)
	embed = get_embedding_matrix(vocab, 'data/sst_glove_6b_300d.txt')
	print(embed(torch.tensor(0)))
	print(embed(torch.tensor(1)))


def pad_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
	texts, labels = zip(*batch)
	lengths = torch.tensor([len(text) for text in texts])
	texts = torch.nn.utils.rnn.pad_sequence(texts, True, 0)
	return texts, torch.tensor(labels, dtype=torch.float).reshape((-1, 1)), lengths
