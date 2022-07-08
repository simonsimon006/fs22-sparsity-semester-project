from torch import rand_like, linspace, sin, pi, Tensor, device
from typing import Tuple


def make_noisy_sine(
    start_time=0,
    end_time=5,
    freq=5,
    sine_amplitude=10,
    noise_amplitude=4,
    time_samples=4096,
    measurements=4096,
    device=device("cpu")) -> Tuple[Tensor, Tensor]:
	'''Creates a noisy sine for training. The sine degrades over the measurements (e.g. to higher vertical indexes). The frequency is expected to be in Hz.'''
	# Create the wanted time.
	time = linspace(start_time, end_time, time_samples, device=device)

	# Produce the sine. Compute the angular frequency from the freq given in Hz.
	signal = sine_amplitude * sin(freq * 2 * pi * time).repeat(measurements, 1)

	# Make the signal become weaker over time to give the sine 2D properties.
	weakening = linspace(0, 1, measurements,
	                     device=device).T.repeat(time_samples, 1)
	weakened_signal = weakening * signal
	print(weakening.shape)
	# Create the random noise (0 to 1), shift it to (-0.5, 0.5) and multiply it
	# by noise_amplitude to get it up to the desired strength. The two is so
	# that the amplitude is as expected and not just the elongation.
	noise = 2 * noise_amplitude * (rand_like(signal, device=device) - 0.5)

	# Create the desired signal.
	data = weakened_signal + noise

	return weakened_signal, data
