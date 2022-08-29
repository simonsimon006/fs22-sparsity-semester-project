"""Loads previously imputed data."""

from typing import Tuple

from numpy import ndarray
from torchvision.datasets import DatasetFolder
from torch import load


class MeasurementFolderLoader(DatasetFolder):
	"""Loads a folder of subfolders with data."""

	# Relevant file extensions
	extensions = ("pt", )

	def __init__(self, root):
		super().__init__(root,
		                 loader=self.__loader__,
		                 extensions=self.extensions)

	def __loader__(self, path: str) -> Tuple[ndarray, ndarray]:
		return load(path)
