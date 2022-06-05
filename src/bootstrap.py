from denoiser import thresh
from numpy import mean, median, count_nonzero, divide, std
from jax.numpy.linalg import svd
import pandas as pd
from loader import MeasurementFolderLoader
from pathlib import Path
from collections import namedtuple
from jax import jit as jax_jit
from jax import random
from functools import partial

data_folder = Path("/home/simon/Code/quelltext/datasets/")

# A data structure to contain the results of one of the svd
# decompositions and their thresholding.
SingularStats = namedtuple("SingularStats",
                           ["SurvivorCounts", "TotalCounts", "Thresholds"])


# No overall jax_jit because of the low-level scipy functions used in thresh.
def get_svd_stat(dataset):

	# How often we sample for a specific size
	TRIES = 10
	PIECES = 10
	SIZES = range(dataset.shape[0] // PIECES, dataset.shape[0],
	              dataset.shape[0] // PIECES)

	t_vals = {}
	# Generates a jax random key
	jax_key = random.PRNGKey(0)
	#rng = default_rng()

	for size in SIZES:
		t_vals[size] = SingularStats([], [], [])
		for _ in range(TRIES):
			print(f"size: {size} of {SIZES}, try: {_+1} of {TRIES}")
			jax_key, singular_values = svd_on_selection(dataset, size, jax_key)
			n, m = (size, dataset.shape[1])
			t = thresh(n, m, singular_values)
			count = count_nonzero(singular_values > t)

			t_vals[size].SurvivorCounts.append(count)
			t_vals[size].TotalCounts.append(len(singular_values))
			t_vals[size].Thresholds.append(t)

	return t_vals


@partial(jax_jit, static_argnums=1)
def svd_on_selection(dataset, size, jax_key):
	# Split the key for usage. We need to use subkey for the selection.
	jax_key, subkey = random.split(jax_key)
	selection = random.choice(
	    key=subkey,
	    a=dataset,
	    replace=False,
	    # That the shape only needs size feels like a jax bug. Maybe I
	    # misunderstood smth..
	    shape=(size, ),
	    axis=0)
	singular_values = svd(selection, compute_uv=False)
	return jax_key, singular_values


dsf = MeasurementFolderLoader(data_folder)

store = []

for (ds, class_index) in dsf:
	print(f"Starting with cat: {class_index}")

	stats = get_svd_stat(ds.numpy())

	for (size, data) in stats.items():
		totalCounts = data.TotalCounts
		survivorCounts = data.SurvivorCounts
		thresholds = data.Thresholds
		survivorCountsRel = divide(survivorCounts, totalCounts)
		store.append(
		    pd.Series({
		        "MaterialCategory": dsf.classes[class_index],
		        "SubsampleSize": size,
		        "ThresholdMean": mean(thresholds),
		        "ThresholdMedian": median(thresholds),
		        "ThresholdStd": std(thresholds),
		        "CountPieces": len(thresholds),
		        "SurvivorCountMean": mean(survivorCounts),
		        "SurvivorCountStd": std(survivorCounts),
		        "RelativeSurvivorCountMean": mean(survivorCountsRel),
		        "RelativeSurvivorCountStd": std(survivorCountsRel),
		        "SingularValuesTotalCount": mean(totalCounts),
		    }))
		print(store[-1])

pd.DataFrame(store).to_csv("bootstrap.csv")
