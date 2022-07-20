from torch import rand_like, linspace, sin, pi, Tensor, device, cartesian_prod
from typing import Tuple


def make_noisy_sine(
    timesteps=4096,
    freq=5,
    sine_amplitude=10,
    noise_amplitude=4,
    spatial_resolution=4096,
    device=device("cpu")) -> Tuple[Tensor, Tensor, Tensor]:
	'''Creates a noisy sine for training. The sine degrades over the measurements (e.g. to higher vertical indexes). The frequency is expected to be in Hz.'''
	# Create the wanted time.
	time = linspace(0, 1, timesteps, device=device)
	space = linspace(0, 1, spatial_resolution, device=device)

	fabric = cartesian_prod(time, space)
	# Produce the sine. Compute the angular frequency from the freq given in Hz.
	signal = sine_amplitude * sin(freq * 2 * pi * space).repeat(timesteps, 1)

	# Make the signal become weaker over time to give the sine 2D properties.

	weakened_signal = (signal.T * time).T

	# Create the random noise (0 to 1), shift it to (-0.5, 0.5) and multiply it
	# by noise_amplitude to get it up to the desired strength. The two is so
	# that the amplitude is as expected and not just the elongation.
	noise = 2 * noise_amplitude * (rand_like(weakened_signal, device=device) -
	                               0.5)

	# Create the desired signal.
	data = weakened_signal + noise

	return weakened_signal, data, fabric
