from itertools import product, islice
from multiprocessing import Pool

import numpy as np
import pywt as wt
from matplotlib import pyplot as plt

from src.loader_old import RawMeasurement, load_folder_raw

FPATH = "/home/simon/Code/quelltext/datasets/"


def process(x):
	return (x[0], wt.wavedec2(x[1], x[0]))

measurements = islice(load_folder_raw(FPATH), 1)
wlist = islice(wt.wavelist(kind='discrete'), 30)
toProcess = map(lambda x: (x[0], x[1].Raw), product(wlist, measurements))

coeffs = map(process, toProcess)

flatCoeffs = list(map(lambda x: (x[0], wt.coeffs_to_array(x[1])[0]), coeffs))

flatCoeffs = np.array(flatCoeffs)
np.save("wv_coeffs", flatCoeffs)
print(flatCoeffs)
