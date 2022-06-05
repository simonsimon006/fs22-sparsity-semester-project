from pathlib import Path
from loader import MeasurementFolderLoader
from despawn import createDeSpaWN
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.losses import MeanAbsoluteError as MAE
from numpy import expand_dims, squeeze
from scipy.io import savemat, loadmat
from plotter import plot, plot2

model1, model2 = createDeSpaWN(kernelsConstraint='CQF', level=10)

data_folder = Path("/home/simon/Code/quelltext/datasets/")

dsf = MeasurementFolderLoader(data_folder)
model1.compile(Adadelta(10), MAE())

for (ds, cat) in dsf:
	ds = ds.numpy()
	ds = expand_dims(ds, axis=-1)
	ds = expand_dims(ds, axis=-1)

	model1.fit(ds, ds, epochs=30)
	den, wav = model1(ds)
	den = squeeze(den)
	wav = squeeze(wav)
	print(den.shape)
	print(wav.shape)
	ds = squeeze(ds)
	plot2(ds, den)
	plot(ds, den, wav, wav, 0, f"plot-{dsf.classes[cat]}.png")
	#savemat(f"../denoised_{cat}.mat", {"E_filt": den})
	exit()