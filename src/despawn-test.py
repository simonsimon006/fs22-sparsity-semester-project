from math import pi
from pathlib import Path
from loader_old import load_train
from despawn import createDeSpaWN
from tensorflow.keras.optimizers.experimental import Adadelta
from tensorflow.keras.losses import MeanSquaredError as MSE
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from plotter import plot, plot2
from tensorflow.data.experimental import load
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.data.experimental import load
from pywt import Wavelet
from tensorflow import linspace, sin, tile
from tensorflow.random import normal
from matplotlib import pyplot as plt
from tensorflow.data import Dataset

WAVELET_NAME = "db20"
KERNEL_INIT = Wavelet(WAVELET_NAME).dec_hi

EPOCHS = 100
LEVELS = 15

LEARNING_RATE = 1e0
#print(f"Length of Kernel: {len(KERNEL_INIT)}")


def single_cut(data):
	#data = data[:data.shape[0] // 2]
	data = tf.expand_dims(data, axis=-1)
	data = tf.expand_dims(data, axis=-1)
	return data


def cutter(noisy, target):
	return [single_cut(x) for x in (noisy, target)]


def bulldozer(noisy, target):
	if tf.reduce_max(noisy) > 500:
		return (noisy, target)
	else:
		return (noisy, tf.zeros(noisy.shape, dtype="float64"))


def condition(noisy, target):
	return noisy.shape == target.shape


def prepare(tf_data):
	tf_data = tf_data.map(cutter).filter(condition).shuffle(
	    1e8, seed=0).prefetch(tf.data.AUTOTUNE)
	return tf_data


AMPLITUDE = 10
TESTING_FREQ = 5 * 2 * pi
time = linspace(0, 5, 3000)
signal = tile([AMPLITUDE * sin(TESTING_FREQ * time)], (500, 1))

noise = normal(signal.shape, dtype="float64")
#noise = zeros_like(signal, device=device)
data = signal + noise

data_l = tf.expand_dims(data, axis=1)
data_l = tf.expand_dims(data_l, axis=1)

signal_l = tf.expand_dims(signal, axis=1)
signal_l = tf.expand_dims(signal_l, axis=1)

model1, model2 = createDeSpaWN(
    kernelsConstraint='Free',
    lossCoeff="l1",
    #kernelInit=KERNEL_INIT,
    #inputSize=4000,
    initHT=1e1,
)

model1.compile(Adadelta(LEARNING_RATE), MSE())

model1.fit(Dataset.from_tensor_slices((data_l, signal_l)).batch(12),
           epochs=EPOCHS)
denoised, _ = model1(data_l)
denoised = tf.squeeze(denoised, axis=0)
plt.plot(time, denoised.numpy()[0, :], label="Denoised")

plt.plot(time, signal.numpy()[0, :], label="Signal", alpha=0.2)
plt.legend()
plt.show()
'''
den, wav = model1(train_data.as_numpy_iterator())
den = tf.squeeze(den)
wav = tf.squeeze(wav)
print(den.shape)
print(wav.shape)
noisy = tf.squeeze(train_data.numpy())
target = tf.squeeze(den)
'''
#plot2(noisy, den)
#plot(noisy, den, wav, wav, 0, f"plot-{dsf.classes[cat]}.png")
#savemat(f"../denoised_{cat}.mat", {"E_filt": den})
