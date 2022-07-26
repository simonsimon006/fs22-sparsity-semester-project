from torch import rand_like, linspace, sin, pi, Tensor, device, cartesian_prod, squeeze, unsqueeze, conj, concat, isnan
from torch.nn.functional import pad, conv2d, conv_transpose2d
from typing import Tuple
from collections import namedtuple
from torch.fft import rfft, irfft
#from torch.nn.functional import conv2d, conv_transpose2d


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


cutscenes = {
    "eps_yl_w3": (770, "left"),
    "eps_yl_k3": (280, "left"),
    "eps_S3-ZG_H03": (500, "left"),
    "eps_S2-ZG_04": (850, "right")
}

# The cutoff starting point has to be deducted. If a number is subtracted it is
# the cutoff value. See Yasmin.
vip = {
    "eps_S2-ZG_04": [(1227 - 850, "erster Riss"), (1796 - 850, "Zweiter Riss"),
                     (15600 - 850, "Bewehrung teilplastisch")],
    "eps_yl_k3": [
        (600 - 280, "elastische Dehnung"),
        (1700 - 280, "vollplastisch 1"),
        (3000 - 280, "vollplastisch 2"),
        #        (4800 - 280, "vollplastisch 3"),
    ],
    "sine-test": [(200, "Sinus")]
}

PadState = namedtuple("PadState", ("vert", "horiz"))


def circ_pad(filter, target_size):
	filt_len = filter.shape[-1]
	to_pad = target_size - filt_len

	repeats = to_pad // filt_len + 2

	filter = filter.repeat((repeats, ))

	return filter


def pad_wrap(input, padding, mode="replicate"):
	input = unsqueeze(input, dim=0)
	input = unsqueeze(input, dim=0)

	input = pad(input, padding, mode=mode)

	input = squeeze(input, dim=0)
	input = squeeze(input, dim=0)

	return input


def convolve(input, filter, transpose=False):
	'''This function is my attempt at using the FFT to do the convolution. It is much faster than the normal convolution function from PyTorch.'''

	# Length is the length of the input. Some problems with that.
	input_length = input.shape[-1]
	filt_len = filter.shape[-1]

	ra = rfft(input)
	rb = rfft(filter, n=input_length)

	# The transpose convolution is equivalent to complex conjugating the
	# fft coefficients because of math.
	if transpose:
		rb = conj(rb)

	res = irfft(input=ra * rb, n=input_length)

	return res


def wrong_convolve(input, filter, transpose=False):
	print(input.shape)
	input = unsqueeze(input, dim=0)
	input = unsqueeze(input, dim=0)

	filter = unsqueeze(filter, dim=0)
	filter = unsqueeze(filter, dim=0)
	filter = unsqueeze(filter, dim=0)
	if transpose:
		result = conv_transpose2d(input, filter)
	else:
		result = conv2d(input, filter, padding="same")

	result = squeeze(result, dim=0)
	result = squeeze(result, dim=0)
	print(result.shape)
	return result