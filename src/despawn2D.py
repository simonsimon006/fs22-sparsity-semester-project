from cmath import pi
from functools import reduce
from math import log2, floor, ceil
from os import unsetenv
from sys import orig_argv
from timeit import repeat

from torch import (Tensor, clone, concat, exp, flip, nn, squeeze, tensor,
                   unsqueeze, zeros, conj)
from torch.fft import irfft, rfft

from torch.nn.functional import conv1d, pad, sigmoid, conv_transpose1d
from torch.nn.parameter import Parameter
from torch.nn import ReflectionPad1d, ReplicationPad1d

NORM = "backward"
PAD_MODE = "circular"


def even_odd(input):
	'''Splits a sequence into even and odd components.'''
	# So we can split the wavelet stuff and the inputs signals with
	# the same function.
	if len(input.shape) == 1:
		return input[::2], input[1::2]
	else:
		return input[:, ::2], input[:, 1::2]


def rfft_two(input1, input2, common_desired_length, normalization="ortho"):
	'''Rffts both inputs zero-padded to the common_desired_length.'''
	return rfft(input1, common_desired_length,
	            norm=normalization), rfft(input2,
	                                      common_desired_length,
	                                      norm=normalization)


def prepare(coeffs, desired_half_length, norm="ortho"):
	'''Prepares the given input for lifting by first splitting them into even and odd parts and then rfft'ing them zero-padded to the desired (half) length.'''
	return rfft_two(*even_odd(coeffs), desired_half_length, normalization=norm)


def sane_circular_pad(input: Tensor, to_pad):
	'''The circular padding function in PyTorch only allows the input to wrap around once. Fascists.'''
	in_len = input.shape[-1]

	# First calculate how often we can repeat the tensors last dimension
	# before we need to pad the rest. The first plus one is because PyTorchs
	# repeat
	# wants to how many times in total the tensor should be there. Thus, one
	# for the original length and then plus how many times we want to add.
	# The second plus one is because we want to "overpad" and then discard the
	# redundant elements. I do not like PyTorchs pad function.
	to_repeat = to_pad // in_len + 1 + 1

	# We create a mask of ones where we then multiply the last entry by
	# to_repeat because .repeat() wants to know which dimensions it shall
	# keep (i.e. entry in the mask == 1) and which to repeat (i.e. entry in the
	# mask > 1).

	repeated = input.repeat((1, to_repeat))

	return repeated[:, :(in_len + to_pad)]


def convolve(input, filter, phase=1, tr=False):
	'''This function is my attempt at using the FFT to do the convolution. It is much faster than the other convolution function further down. But a time shift occurs if using this one. I vaguely know why this might be, padding and stuff, but I am not (yet) willing to figure it out in detail and fix it.'''

	# Length is the length of the input. Some problems with that.
	input_length = input.shape[-1]
	filter_length = filter.shape[-1]
	'''if input_length < filter_length:
		input = sane_circular_pad(input, filter_length - input_length)

		ra = rfft(input)
		rb = rfft(filter)
	else:'''
	ra = rfft(input)
	if tr:
		rb = conj(rfft(filter, n=input_length))
	else:
		rb = rfft(filter, n=input_length)

	res = irfft(input=ra * rb, n=input_length)

	return res


def fconvolve(input, filter, tr=False, **kwargs):
	'''Basic convolution with (not yet) circular padding.'''

	original_len = input.shape[-1]

	# Padding the input so the convolution returns as many coefficients
	# as there were entries in the input before. The "same" padding of the conv
	# does not work nicely.

	input = sane_circular_pad(input, filter.shape[-1])

	input = unsqueeze(input, dim=1)
	filter = unsqueeze(unsqueeze(filter, dim=0), dim=0)

	if tr:
		res = squeeze(conv_transpose1d(input, filter), 1)
	else:
		res = squeeze(conv1d(input, filter), 1)

	return res[:, :original_len]


class ForwardTransformLayer(nn.Module):
	'''Expects a tensor with shape (n, m) and computes one forward step over each row, i.e. it iterates over all values of 0...n and then computes the lifting. input_length needs to be equal to m.'''

	def __init__(self, scaling):
		super(ForwardTransformLayer, self).__init__()

		# Set normalization mode for rfft and irfft.
		self.norm = NORM

		# Store the scaling function filter.
		self.scaling = scaling

		# Applies an approx. 45 deg phase shift to compensate for that in the
		# output.
		self.phase_correction = exp(tensor(1j * 0.785398))

	def compute_wavelet(self, scaling: Tensor) -> Tensor:
		'''Computes the wavelet filter coefficients from the scaling coefficients. The formula is g_k = h_(M-k-1) where M is the filter length of h. The formula stems from a paper.'''

		# flip copies / clones the input.
		g = flip(scaling, (-1, ))
		g[1::2] *= -1

		return g

	def forward(self, input):

		wavelet = self.compute_wavelet(self.scaling)
		'''
		# It is expected that the inputs rows have the length equal to a power
		# of two.
		part_length = input.shape[1] // 2

		# Splits the filter coefficients into even and odd parts and
		# transforms the wavelet and scaling coefficients.
		# Zero-padding happens in prepare_coefficients.
		wavelet_even, wavelet_odd = even_odd(wavelet)
		scaling_even, scaling_odd = even_odd(self.scaling)

		# Split the input into even and odd.
		even, odd = even_odd(input)

		# Create the detail coefficients.
		details = convolve(
		    even, wavelet_even, norm=self.norm, length=part_length) + convolve(
		        odd, wavelet_odd, norm=self.norm, length=part_length)

		# Filter out the approximation coefficients.
		approximation = convolve(
		    even, scaling_even, norm=self.norm, length=part_length) + convolve(
		        odd, scaling_odd, norm=self.norm, length=part_length)

		if len(details.shape) == 1:
			details = unsqueeze(details, dim=-1)
			approximation = unsqueeze(approximation, dim=-1)

		return details, approximation
		'''

		details = convolve(input, wavelet, phase=self.phase_correction)
		approximation = convolve(input,
		                         self.scaling,
		                         phase=self.phase_correction)

		return details[:, ::2], approximation[:, ::2]


class BackwardTransformLayer(nn.Module):
	'''Expects a tensor with shape (n, m) and computes one backward step over each row, i.e. it iterates over all values of 0...n and then computes the lifting. input_length needs to be equal to m.'''

	def __init__(self, scaling):
		super(BackwardTransformLayer, self).__init__()

		# Set normalization mode for rfft and irfft.
		self.norm = NORM

		# Store the filters.
		self.scaling = scaling

	def compute_wavelet(self, scaling: Tensor) -> Tensor:
		'''Computes the wavelet filter coefficients from the scaling coefficients. The formula is g_k = h_(M-k-1) where M is the filter length of h. The formula stems from a paper.'''

		# flip copies / clones the input.
		g = flip(scaling, (-1, ))
		g[1::2] *= -1

		return g

	# These are required for biorthogonal wavelets
	def compute_scaling_synthesis(self, scaling: Tensor) -> Tensor:
		return flip(scaling, (-1, ))

	def compute_wavelet_synthesis(self, scaling: Tensor) -> Tensor:
		# We have to clone it so we do not change the wavelet tensor for
		# the rest of the class.
		g_hat = clone(scaling)
		g_hat[::2] *= -1

		return g_hat

	def forward(self, input):

		# Compute synthesis filter coefficients
		wavelet_synthesis = self.compute_wavelet(self.scaling)
		scaling_synthesis = self.scaling

		# Pad the input to an even size.
		(details, approximation) = input
		# We are creating new tensors so we have to tell them how they should look
		device = details.device

		# Interpolate with zeros.
		filled_details = zeros(details.shape[0],
		                       details.shape[1] * 2,
		                       device=device)
		filled_approximation = zeros(approximation.shape[0],
		                             approximation.shape[1] * 2,
		                             device=device)
		filled_details[:, ::2] = details
		filled_approximation[:, ::2] = approximation

		# Get the lengths of the signals.
		fil_details_len = filled_details.shape[1]
		fil_approx_len = filled_approximation.shape[1]

		# Convolve with the reconstruction filters.
		details_rec = convolve(filled_details, wavelet_synthesis, tr=True)
		approx_rec = convolve(filled_approximation, scaling_synthesis, tr=True)

		return details_rec + approx_rec


class HardThreshold(nn.Module):

	def __init__(self, alpha=10, b_plus=2, b_minus=1):
		super(HardThreshold, self).__init__()
		self.alpha = Parameter(alpha)
		self.b_plus = Parameter(b_plus)
		self.b_minus = Parameter(b_minus)

	def denominator(self, x, alpha, bias):
		res = 1. + exp(alpha * (x + bias))
		return res

	def forward(self, input):
		return input * (sigmoid(self.alpha * (input - self.b_plus)) +
		                sigmoid(-self.alpha * (input + self.b_minus)))

	'''
	def forward(self, input):
		return multiply(
		    input, ((1. / self.denominator(input, self.alpha, self.b_minus)) +
		            (1. / self.denominator(input, -self.alpha, -self.b_plus))))
	'''


class Despawn2D(nn.Module):
	'''Expects the data to come in blocks with shape (n, m). The blocks can have different shapes but the rows of the blocks have to have constant length within a block.'''

	def __init__(self,
	             levels: int,
	             scaling: Tensor,
	             adapt_filters: bool = True):
		super(Despawn2D, self).__init__()
		self.scaling = Parameter(scaling.repeat(levels, 1),
		                         requires_grad=adapt_filters)

		self.levels = levels

		# Define forward transforms
		self.forward_transforms = [
		    ForwardTransformLayer(self.scaling[level, :])
		    for level in range(self.levels)
		]

		# Define threshold parameters similar to the paper
		self.alpha = tensor(5.)
		self.b_plus = tensor(2.)
		self.b_minus = tensor(1.)

		# Define hard thresholding layer
		self.threshold = HardThreshold(self.alpha, self.b_plus, self.b_minus)

		# Define backward transformations
		self.backward_transforms = [
		    BackwardTransformLayer(self.scaling[level, :])
		    for level in reversed(range(self.levels))
		]

	def forward(self, input):
		# Pad the input on the second axis to the length of a power of two.
		input_length = input.shape[1]

		desired_input_length = 2**(ceil(log2(input_length)))

		# Pads the input if the input size is not a power of two.
		# The circular padding is necessary for perfect reconstruction.
		# And it gives the least amount of coefficients.
		'''input = sane_circular_pad(
		    input,
		    desired_input_length - input_length,
		)'''
		to_pad = ((desired_input_length - input_length) // 2,
		          ceil((desired_input_length - input_length) / 2))
		padder = ReplicationPad1d(to_pad)
		input = padder(input)

		approximation = input
		wav_coeffs = []
		for transform in self.forward_transforms:
			details, approximation = transform(approximation)
			wav_coeffs.append(details)
		wav_coeffs.append(approximation)

		# Filter the coefficients.
		#filtered = list(map(self.threshold, wav_coeffs))
		filtered = list(wav_coeffs)
		# Transform the filtered inputs back.
		approximation = filtered.pop()
		for transform in self.backward_transforms:
			details = filtered.pop()
			approximation = transform((details, approximation))

		return approximation[:, to_pad[0]:input_length + to_pad[1]], reduce(
		    lambda tensor1, tensor2: concat(tensors=(tensor1, tensor2), dim=-1),
		    wav_coeffs)
