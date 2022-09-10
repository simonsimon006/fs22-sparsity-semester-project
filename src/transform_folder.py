from loader_matlab import load_train
import torch as t

import pathlib as pl
from joblib import Parallel, delayed

FOLDER_PATH = pl.Path("/home/simon/Code/quelltext/datasets/")
STORE_DIR = pl.Path("store/")
files = FOLDER_PATH.glob("*.mat")


def process(file):
	filename = file.stem
	print(f"Loading file {filename}")
	meas = load_train(file)
	print(f"Finished loading file {filename}")

	t.save((meas.Filled, meas.Filtered), STORE_DIR / f"{filename}.pt")


Parallel(n_jobs=6, batch_size=1)(delayed(process)(file) for file in files)
