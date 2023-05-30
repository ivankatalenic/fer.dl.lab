import torch
import torch.nn as nn
import torch.nn.functional as F


class _BNReluConv(nn.Sequential):
	def __init__(self, num_maps_in: int, num_maps_out: int, k: int = 3, bias: bool = True) -> None:
		super().__init__()
		self.append(nn.BatchNorm2d(num_features=num_maps_in))
		# Should a ReLU activation really come before a convolution layer?
		self.append(nn.ReLU())
		self.append(nn.Conv2d(in_channels=num_maps_in, out_channels=num_maps_out, kernel_size=k, bias=bias))

class _AvgPoolAll(torch.nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		"""
		Input: N x C x H x W
		Output: N x C
		"""
		return torch.mean(input, dim=(2, 3))

class SimpleMetricEmbedding(nn.Module):
	def __init__(self, input_channels: int, emb_size: int = 32) -> None:
		super().__init__()
		self.emb_size = emb_size
		self.stack = nn.Sequential(
			_BNReluConv(input_channels, emb_size),
			nn.MaxPool2d(kernel_size=3, stride=2),
			_BNReluConv(emb_size, emb_size),
			nn.MaxPool2d(kernel_size=3, stride=2),
			_BNReluConv(emb_size, emb_size),
			_AvgPoolAll()
		)

	def get_features(self, img: torch.Tensor) -> torch.Tensor:
		# Returns tensor with dimensions BATCH_SIZE, EMB_SIZE
		x = self.stack(img)
		return x

	def loss(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
		a_x = self.get_features(anchor)
		p_x = self.get_features(positive)
		n_x = self.get_features(negative)
		
		d_ap = F.pairwise_distance(a_x, p_x, keepdim=True)
		d_an = F.pairwise_distance(a_x, n_x, keepdim=True)
		margin = 1.0
		loss = torch.mean(torch.clamp(d_ap - d_an + margin, min=0))
		return loss
