'''Provides functions to load and preprocess the data.'''
import pathlib as pl
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor as PPE
from typing import Dict, Generator, Tuple, Union

import numpy as np
import scipy.io
import tensorflow as tf
from numba import jit
from numpy import ceil, concatenate, floor, ndarray
from sklearn.impute import KNNImputer

MAX_SIZE = 8192
RawMeasurement = namedtuple("RawMeasurement", ["Raw", "Filtered"])
FilledMeasurement = namedtuple("FilledMeasurement", ["Filled", "Filtered"])
PaddedMeasurement = namedtuple("PaddedMeasurement", ["Padded", "Filtered"])


@jit(parallel=True, nogil=True)
def padder(array: ndarray, end_width: int = MAX_SIZE) -> ndarray:
	'''Symmetrically padds the rows of a 2d array to the end_width size. Uses 
	zeros.'''
	# Calculate how much we need to pad.
	to_pad = (end_width - array.shape[1]) / 2
	if to_pad < 0:
		raise ValueError(
		    f"MAX_SIZE: {MAX_SIZE} is smaller than axis one of the given array."
		)
	lower = int(floor(to_pad))
	upper = int(ceil(to_pad))
	res = np.pad(array,
	             pad_width=(
	                 (0, 0),
	                 (lower, upper),
	             ),
	             constant_values=(0, ))
	return res


@jit(parallel=True, nogil=True)
def other_padder(array: ndarray, end_width: int = MAX_SIZE) -> ndarray:
	'''Symmetrically padds the rows of a 2d array to the end_width size. Uses 
	zeros.'''
	to_pad = (end_width - array.shape[1]) / 2
	up = np.ceil(to_pad)
	down = np.floor(to_pad)
	if to_pad < 0:
		raise ValueError(
		    f"MAX_SIZE: {MAX_SIZE} is smaller than axis one of the given array."
		)
	new = np.pad(array, ((0, 0), (down, up)))
	return new


def replace_nan(array: ndarray) -> ndarray:
	'''Replaces NaN in the given dataset.'''
	imputer = KNNImputer(n_neighbors=2, weights="distance", copy=True)
	return imputer.fit_transform(array)


@jit(parallel=True)
def load_train(path: str) -> Tuple[ndarray, ndarray]:
	'''Loads the filled and filtered dataset from the matlab file under the specified path.'''
	data: Dict[str, ndarray] = scipy.io.loadmat(
	    path, variable_names=["E_filled", "E_filt"])
	return data["E_filled"], data["E_filt"]


@jit(parallel=True, nogil=True)
def load_raw(path: Union[str, pl.Path]) -> RawMeasurement:
	'''Loads the raw (unfilled) and filtered parts of a matlab data file.'''
	data: Dict[str,
	           ndarray] = scipy.io.loadmat(path,
	                                       variable_names=["E_raw", "E_filt"])
	return RawMeasurement(data["E_raw"], data["E_filt"])


@jit(parallel=True, nogil=True)
def just_raw(path):
	'''Loads only the raw (unfilled) part of a matlab data file.'''
	return scipy.io.loadmat(path, variable_names=["E_raw"])["E_raw"]


def load_folder_raw(
        path: Union[str, pl.Path]) -> Generator[RawMeasurement, None, None]:
	'''Used to load all the weird matlab files in a folder given by path. Only looks at files in the first level of the directory.'''
	p = pl.Path(path)
	data = (load_raw(measurement) for measurement in p.iterdir()
	        if measurement.is_file())
	return data


def make_tf_dataset(
        measurements: Generator[RawMeasurement, None, None]) -> tf.data.Dataset:
	'''This function transforms a folder of the weird matlab files, represented by measurements, into one big, imputed and padded TensorFlow dataset.'''
	# Impute the NaN values.
	filled = (FilledMeasurement(replace_nan(measurement.Raw),
	                            measurement.Filtered)
	          for measurement in measurements)
	# Pad them to MAX_SIZE.
	padded = (PaddedMeasurement(padder(measurement.Filled, MAX_SIZE),
	                            padder(measurement.Filtered, MAX_SIZE))
	          for measurement in filled)

	# Disentangle the tuples
	pad = [elem.Padded for elem in padded]
	filt = [elem.Filtered for elem in padded]

	# Concatenate them.
	flat_pad = concatenate(pad, axis=0)
	flat_filt = concatenate(filt, axis=0)

	# Make them into a dataset.
	return tf.data.Dataset.from_tensor_slices((flat_pad, flat_filt))


def make_np_dataset(
    measurements: Generator[RawMeasurement, None,
                            None]) -> Tuple[ndarray, ndarray]:
	'''This function transforms a folder of the weird matlab files, represented by measurements, into two big, imputed and padded numpy arrays.'''
	# Impute the NaN values.
	filled = (FilledMeasurement(replace_nan(measurement.Raw),
	                            measurement.Filtered)
	          for measurement in measurements)
	# Pad them to MAX_SIZE.
	padded = (PaddedMeasurement(padder(measurement.Filled, MAX_SIZE),
	                            padder(measurement.Filtered, MAX_SIZE))
	          for measurement in filled)

	# Disentangle the tuples
	with PPE() as pool:
		pad = pool.map(lambda elem: elem.Padded, padded)
		filt = pool.map(lambda elem: elem.Filtered, padded)

		#pad = (elem.Padded for elem in padded)
		#filt = (elem.Filtered for elem in padded)

		# Concatenate them.
		flat_pad = concatenate(pad, axis=0)
		flat_filt = concatenate(filt, axis=0)

	# Make them into a dataset.
	return ndarray(flat_pad), ndarray(flat_filt)
