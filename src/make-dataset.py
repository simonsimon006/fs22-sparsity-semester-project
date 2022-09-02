from functools import reduce
from loader import MeasurementFolder
from tensorflow.data import Dataset
from tensorflow.io import TFRecordWriter
from multiprocessing import Pool
from loader_old import make_tf_dataset, load_folder_raw
from tensorflow.config import set_visible_devices
from tensorflow.data.experimental import save

DATA_PATH = "concrete/"
SET_PATHS = "/home/simon/Code/quelltext/datasets/"
set_visible_devices([], 'GPU')
#dsf = MeasurementFolderLoader(SET_PATHS)
'''
def worker(a):
	(f, c) = a
	print(f"doing a thing of class {c}")
	d = Dataset.from_generator(zip(*[x.numpy().T for x in f]))
	return d


with Pool(processes=6) as p:
	datasets = p.map(worker, dsf)

ds = reduce(lambda old, new: old.concatenate(new), datasets)
'''

folder_generator = load_folder_raw(SET_PATHS)
ds = make_tf_dataset(folder_generator)
save(ds, DATA_PATH)
'''
with TFRecordWriter("concrete_set.tfrecord") as writer:
	writer.write(ds)'''