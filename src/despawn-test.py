from pathlib import Path
from loader import MeasurementFolderLoader
from despawn import createDeSpaWN
from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.losses import MeanSquaredError as MSE

from plotter import plot, plot2
from tensorflow.data.experimental import load
import tensorflow as tf

PREFIX = "/home/simon/Code/quelltext/"
TRAIN_PATH = PREFIX + "train/"
VALIDATION_PATH = PREFIX + "validation-4200/"
BS = 512
PADD = 4096
LEVELS = 14
# Daub 2 low pass
KERNEL_INIT = [
    float(x) for x in '''-0.12940952255092145
0.22414386804185735
0.836516303737469
0.48296291314469025'''.split("\n")
]
print(len(KERNEL_INIT))


def single_cut(data):
	data = data[:data.shape[0] // 2]
	data = tf.expand_dims(data, axis=-1)
	data = tf.expand_dims(data, axis=-1)
	return data


def cutter(noisy, target):
	return [single_cut(x) for x in (noisy, target)]


def prepare(tf_data):
	tf_data = tf_data.filter(lambda x, y: tf.reduce_max(
	    x) > 500 and tf.reduce_max(y) > 500).map(cutter).batch(BS).prefetch(
	        tf.data.AUTOTUNE)
	return tf_data


train_data = prepare(load(TRAIN_PATH, compression="GZIP"))
validation_data = prepare(load(VALIDATION_PATH, compression="GZIP"))

model1, model2 = createDeSpaWN(
    kernelsConstraint='CQF',
    level=LEVELS,
    lossCoeff="l1",
    kernelInit=KERNEL_INIT,
    inputSize=4000,
    #initHT=0,
)

data_folder = Path("/home/simon/Code/quelltext/datasets/")

#dsf = MeasurementFolderLoader(data_folder)
model1.compile(Adadelta(1), MSE())

#for (ds, cat) in dsf:

model1.fit(train_data, epochs=100, validation_data=validation_data)
'''den, wav = model1(noisy)
	den = squeeze(den)
	wav = squeeze(wav)
	print(den.shape)
	print(wav.shape)
	noisy = squeeze(noisy)
	target = squeeze(target)'''

#plot2(noisy, den)
#plot(noisy, den, wav, wav, 0, f"plot-{dsf.classes[cat]}.png")
#savemat(f"../denoised_{cat}.mat", {"E_filt": den})
