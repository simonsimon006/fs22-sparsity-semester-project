"""Implements loading and imputing the data"""

from typing import Dict

from numpy import ndarray
from scipy.io import loadmat
from sklearn.impute import KNNImputer
from torchvision.datasets import DatasetFolder
from torch import Tensor, from_numpy


def __transform__(measurement: ndarray) -> Tensor:
	"""Transforms the image to a tensor and applies imputing."""
	imputer = KNNImputer(n_neighbors=3, weights="distance", copy=False)
	measurement = imputer.fit_transform(measurement)
	return from_numpy(measurement)


class MeasurementFolderLoader(DatasetFolder):
	"""Loads a folder of subfolders with data. The structure has to be
	root/CLASS/measurement.mat where class is e.g. reinforced concrete."""

	# Which entry from the .mat dict to load
	__entryToLoad = "E_raw"

	# Relevant file extensions
	extensions = ("mat", )

	def __init__(self, root):
		super().__init__(root,
		                 loader=self.__loader__,
		                 transform=__transform__,
		                 extensions=self.extensions)

	def __loader__(self, path: str) -> ndarray:
		file: Dict[str, ndarray] = loadmat(path,
		                                   variable_names=self.__entryToLoad,
		                                   matlab_compatible=True)
		return file["E_raw"]
