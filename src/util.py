from collections import namedtuple
from functools import reduce
from typing import Tuple

from torch import (Tensor, abs, cartesian_prod, conj, device, flip, linspace,
                   mean, pi, rand_like, sin, squeeze, unsqueeze)
from torch.fft import irfft, rfft
from torch.nn.functional import pad


def make_noisy_sine(
    timesteps=4096,
    freq=5,
    sine_amplitude=10,
    noise_amplitude=4,
    spatial_resolution=4096,
    dev=device("cpu")) -> Tuple[Tensor, Tensor, Tensor]:
	'''Creates a noisy sine for training. The sine degrades over the measurements (e.g. to higher vertical indexes). The frequency is expected to be in Hz.'''
	# Create the wanted time.
	time = linspace(0, 1, timesteps, device=dev)
	space = linspace(0, 1, spatial_resolution, device=dev)

	fabric = cartesian_prod(time, space)
	# Produce the sine. Compute the angular frequency from the freq given in Hz.
	signal = sine_amplitude * sin(freq * 2 * pi * space).repeat(timesteps, 1)

	# Make the signal become weaker over time to give the sine 2D properties.

	weakened_signal = (signal.T * time).T

	# Create the random noise (0 to 1), shift it to (-0.5, 0.5) and multiply it
	# by noise_amplitude to get it up to the desired strength. The two is so
	# that the amplitude is as expected and not just the elongation.
	noise = 2 * noise_amplitude * (rand_like(weakened_signal, device=dev) - 0.5)

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


def l1_reg(coeffs):
	if type(coeffs) == tuple or type(coeffs) == list:
		return reduce(lambda acc, elem: acc + mean(abs(elem)), coeffs, 0)
	else:
		return mean(abs(coeffs))


def compute_wavelet(scaling: Tensor) -> Tensor:
	'''Computes the wavelet filter coefficients from the scaling coefficients. The formula is g_k = h_(M-k-1) where M is the filter length of h. The formula stems from a paper.'''

	# flip copies / clones the input.
	g = flip(scaling, (-1, ))
	g[1::2] *= -1

	return g


# Only required for biorthogonal instead of orthogonal wavelets.
def compute_scaling_synthesis(scaling: Tensor) -> Tensor:
	return flip(scaling, (-1, ))


def compute_wavelet_synthesis(scaling: Tensor) -> Tensor:
	# flip copies / clones the input.
	g = flip(scaling, (-1, ))
	g[::2] *= -1

	return g


def convolve(input, filter, transpose=False):
	'''This function is my attempt at using the FFT to do the convolution. It is much faster than the normal convolution function from PyTorch.'''

	# Length is the length of the input. Some problems with that.
	input_length = input.shape[-1]

	ra = rfft(input)
	rb = rfft(filter, n=input_length)

	# The transpose convolution is equivalent to complex conjugating the
	# fft coefficients because of math.
	if transpose:
		rb = conj(rb)

	res = irfft(input=ra * rb, n=input_length)

	return res


def convolve_downsample(input, filter):
	'''Convolves and decimates the output size by two. So we do not compute the unnecessary coefficients.'''
	input_even = input[:, ::2]
	input_odd = input[:, 1::2]

	filter_even = filter[::2]
	filter_odd = filter[1::2]

	even = convolve(input_even, filter_even)
	odd = convolve(input_odd, filter_odd)

	return even + odd
