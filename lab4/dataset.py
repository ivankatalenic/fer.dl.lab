from torch.utils.data import Dataset
from collections import defaultdict
from random import choice
from typing import Literal, Optional
import torchvision
import torch

class MNISTMetricDataset(Dataset):
	def __init__(
			self,
			root: str = "/tmp/mnist/",
			split: Literal['train', 'test', 'traineval'] = 'train',
			remove_class: Optional[int] = None
		) -> None:
		super().__init__()
		assert split in ['train', 'test', 'traineval']
		self.root = root
		self.split = split

		mnist_ds = torchvision.datasets.MNIST(self.root, train='train' in split, download=True)
		self.images, self.targets = mnist_ds.data.float() / 255., mnist_ds.targets
		self.classes = list(range(10))

		if remove_class is not None:
			self.classes.remove(remove_class)
			valid = self.targets != remove_class
			self.images = self.images[valid]
			self.targets = self.targets[valid]

		self.target2indices = defaultdict(list)
		for i in range(len(self.images)):
			self.target2indices[self.targets[i].item()] += [i]

	def _sample_negative(self, index: int) -> int:
		target = self.targets[index].item()
		possible_targets = set(self.classes)
		possible_targets.discard(target)
		other_target = choice(list(possible_targets))
		return choice(self.target2indices[other_target])

	def _sample_positive(self, index: int) -> int:
		target = self.targets[index].item()
		return choice(self.target2indices[target])

	def __getitem__(self, index: int) -> tuple[torch.Tensor, int] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
		anchor = self.images[index].unsqueeze(0)
		target_id = self.targets[index].item()
		if self.split in ['traineval', 'val', 'test']:
			return anchor, target_id
		
		positive = self._sample_positive(index)
		negative = self._sample_negative(index)
		positive = self.images[positive]
		negative = self.images[negative]
		return anchor, positive.unsqueeze(0), negative.unsqueeze(0), target_id

	def __len__(self) -> int:
		return len(self.images)
