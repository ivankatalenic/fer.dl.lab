import torch

class IdentityModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def get_features(self, img: torch.Tensor) -> torch.Tensor:
        return img.flatten(start_dim=1)
