"""Loads previously imputed data."""

from typing import Tuple
import pathlib as pl

from torch import load, Tensor, float32, tensor
from torch.utils.data import IterableDataset


class MeasurementFolder(IterableDataset):
	"""Loads a folder of subfolders with data."""

	def __init__(self, root):
		super().__init__()
		root = pl.Path(root)
		self.files = list(root.glob("*.pt"))

	def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
		if index not in self.files:
			raise ValueError(
			    f"{index} is not a filename for an available measurement.")
		else:
			return self.__load(index)

	def __load(self, path):
		input, target = load(path)

		return (tensor(input, dtype=float32), tensor(target, dtype=float32))

	def __len__(self):
		return len(self.files)

	def __iter__(self):
		return map(self.__load, self.files)