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


'''
train_data = prepare(
    tf.data.Dataset.from_tensor_slices(
        load_train(PREFIX +
                   TRAIN_SET)))  #prepare(load(TRAIN_PATH, compression="GZIP"))
validation_data = prepare(
    tf.data.Dataset.from_tensor_slices(load_train(
        PREFIX + VAL_SET)))  #prepare(load(VALIDATION_PATH, compression="GZIP"))
'''
'''
'''
#train_data = load("dataset")
train_data = prepare(load("steelbar"))
rlp = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-5)
'''
ax = set()
ay = set()

for (x, y) in train_data:
	ax.add(x.shape)
	ay.add(y.shape)

print(ax)
print(ay)
#exit()
'''
model1, model2 = createDeSpaWN(
    kernelsConstraint='Free',
    lossCoeff="l1",
    kernelInit=KERNEL_INIT,
    #inputSize=4000,
    initHT=1e2,
)

model1.compile(Adadelta(LEARNING_RATE), MSE())

model1.fit(
    train_data,
    epochs=EPOCHS,
    #validation_data=validation_data,
    callbacks=[rlp])
model1.save("model1")
model2.save("model2")
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
