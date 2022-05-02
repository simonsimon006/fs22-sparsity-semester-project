"""Implements loading and imputing the data"""

import pathlib
from typing import Dict

from numpy import ndarray
from scipy.io import loadmat
from sklearn.impute import KNNImputer
from torchvision.datasets import DatasetFolder
from torch import Tensor, from_numpy

cutscenes = {
    "eps_yl_w3": 770,
    "eps_yl_k3": 280,
    "eps_S3-ZG_H03": 500,
    "eps_S2-ZG_04": 850
}


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
		parsed_path = pathlib.Path(path)
		filename = parsed_path.stem
		entry = file[self.__entryToLoad]

		# Ignore the redundant second half (spatial, i.e. over the x axis) of
		# the signal.
		half_index = (entry.shape[1] // 2)
		entry = entry[:, :half_index]

		# If we now when the actual measurement for the file begins we start
		# with that timestep.
		if filename in cutscenes:
			starting_time = cutscenes[filename]
			return entry[starting_time:, :]
		else:
			return entry
