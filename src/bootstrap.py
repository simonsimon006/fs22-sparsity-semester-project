from tokenize import single_quoted
#from scipy.io import loadmat
from denoiser import thresh
from numpy.linalg import svd
from numpy.random import Generator, default_rng
from numpy import empty, mean, median, std
import pandas as pd
from loader import MeasurementFolderLoader
from pathlib import Path

data_folder = Path("/home/simon/Code/quelltext/datasets/")

#filename = "/home/simon/Code/quelltext/datasets/eps_yl_k3.mat"
#ds = loadmat(filename, variable_names=["E_filled"])
#fill = ds["E_filled"]


def get_svd_stat(dataset):

	# How often we sample for a specific size
	TRIES = 2
	PIECES = 10
	SIZES = range(dataset.shape[0] // PIECES, dataset.shape[0] // 2,
	              dataset.shape[0] // PIECES)

	t_vals = {}

	rng = default_rng()

	for size in SIZES:
		t_vals[size] = []
		for _ in range(TRIES):
			print(f"size: {size} of {SIZES}, try: {_} of {TRIES}")
			selection = rng.choice(dataset,
			                       replace=False,
			                       shuffle=False,
			                       size=size)
			singular_values = svd(selection, compute_uv=False)
			n, m = selection.shape
			t = thresh(n, m, singular_values)
			t_vals[size].append(t[0])

	return t_vals


dsf = MeasurementFolderLoader(data_folder)

store = []

for (ds, cat) in dsf:
	print(f"Starting with cat: {cat}")

	stats = get_svd_stat(ds)

	for (size, data) in stats.items():
		store.append(
		    pd.Series({
		        "category": cat,
		        "size": size,
		        "mean": mean(data),
		        "median": median(data),
		        "std": std(data)
		    }))
	break

pd.DataFrame(store).to_csv("bootstrap.csv")
