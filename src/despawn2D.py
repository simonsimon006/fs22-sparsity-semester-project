from functools import reduce
from math import ceil, log2

from torch import (Tensor, clone, concat, divide, exp, flip, multiply, nn,
                   squeeze, tensor, unsqueeze, zeros)
from torch.fft import irfft, rfft

from torch.nn.functional import conv1d
from torch.nn.parameter import Parameter

NORM = "backward"


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


def convolve(a, b, norm, length):
	ra = rfft(a, norm=norm, n=length)
	rb = rfft(b, norm=norm, n=length)
	return irfft(input=ra * rb, norm=norm, n=length)


def fconvolve(a, b, **kwargs):
	a = unsqueeze(a, dim=1)
	b = unsqueeze(unsqueeze(b, dim=0), dim=0)
	return squeeze(conv1d(a, b, padding="same"))


class ForwardTransformLayer(nn.Module):
	'''Expects a tensor with shape (n, m) and computes one forward step over each row, i.e. it iterates over all values of 0...n and then computes the lifting. input_length needs to be equal to m.'''

	def __init__(self, wavelet):
		super(ForwardTransformLayer, self).__init__()

		# Set normalization mode for rfft and irfft.
		self.norm = NORM

		# Store the wavelet and scaling function filters.
		self.wavelet = wavelet

	def compute_scaling(self, wavelet: Tensor) -> Tensor:
		'''Computes the scaling filter coefficients from the wavelet coefficients. The formula is g_k = h_(M-k-1) where M is the filter length of h. The formula stems from a paper.'''

		# flip copies / clones the input.
		g = flip(wavelet, (-1, ))
		g[1::2] *= -1

		return g

	def forward(self, input):
		# It is expected that the inputs rows have the length equal to a power
		# of two.
		self.part_length = input.shape[1] // 2
		scaling = self.compute_scaling(self.wavelet)

		# Splits the filter coefficients into even and odd parts and
		# transforms the wavelet and scaling coefficients.
		# Zero-padding happens in prepare_coefficients.
		wavelet_even, wavelet_odd = even_odd(self.wavelet)
		scaling_even, scaling_odd = even_odd(scaling)

		# Split the input into even and odd.
		even, odd = even_odd(input)

		# Create the detail coefficients.
		details = convolve(
		    even, wavelet_even, norm=self.norm,
		    length=self.part_length) + convolve(
		        odd, wavelet_odd, norm=self.norm, length=self.part_length)

		# Filter out the approximation coefficients.
		approximation = convolve(
		    even, scaling_even, norm=self.norm,
		    length=self.part_length) + convolve(
		        odd, scaling_odd, norm=self.norm, length=self.part_length)

		return details, approximation
		'''
		details = convolve(input,
		                   self.wavelet,
		                   norm=self.norm,
		                   length=input.shape[1])
		approximation = convolve(input,
		                         scaling,
		                         norm=self.norm,
		                         length=input.shape[1])
		return details[:, ::2], approximation[:, ::2]
		'''


class BackwardTransformLayer(nn.Module):
	'''Expects a tensor with shape (n, m) and computes one backward step over each row, i.e. it iterates over all values of 0...n and then computes the lifting. input_length needs to be equal to m.'''

	def __init__(self, wavelet):
		super(BackwardTransformLayer, self).__init__()

		# Set normalization mode for rfft and irfft.
		self.norm = NORM

		# Store the filters.
		self.wavelet = wavelet

	def compute_wavelet_synthesis(self, wavelet: Tensor) -> Tensor:
		return flip(wavelet, (-1, ))

	def compute_scaling_synthesis(self, wavelet: Tensor) -> Tensor:
		# We have to clone it so we do not change the wavelet tensor for
		# the rest of the class.
		g_hat = clone(wavelet)
		g_hat[1::2] *= -1

		return g_hat

	def forward(self, input):

		# Compute synthesis filter coefficients
		wavelet_synthesis = self.compute_wavelet_synthesis(self.wavelet)
		scaling_synthesis = self.compute_scaling_synthesis(self.wavelet)

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
		details_rec = convolve(filled_details,
		                       wavelet_synthesis,
		                       norm=self.norm,
		                       length=fil_details_len)
		approx_rec = convolve(filled_approximation,
		                      scaling_synthesis,
		                      norm=self.norm,
		                      length=fil_approx_len)

		return details_rec + approx_rec


class HardThreshold(nn.Module):

	def __init__(self, alpha=10, b_plus=1, b_minus=1):
		super(HardThreshold, self).__init__()
		self.alpha = Parameter(alpha)
		self.b_plus = Parameter(b_plus)
		self.b_minus = Parameter(b_minus)

	def denominator(self, x, alpha, bias):
		return 1 + exp((alpha * x + bias))

	def forward(self, input):
		return multiply(
		    input,
		    divide(1, self.denominator(input, self.alpha, self.b_plus)) +
		    divide(1, self.denominator(input, -self.alpha, self.b_minus)))


class Despawn2D(nn.Module):
	'''Expects the data to come in blocks with shape (n, m). The blocks can have different shapes but the rows of the blocks have to have constant length within a block.'''

	def __init__(self,
	             levels: int,
	             wavelet: Tensor,
	             adapt_filters: bool = True):
		super(Despawn2D, self).__init__()
		self.wavelet = Parameter(wavelet.repeat(levels, 1),
		                         requires_grad=adapt_filters)

		self.levels = levels

		# Define forward transforms
		self.forward_transforms = [
		    ForwardTransformLayer(self.wavelet[level, :])
		    for level in range(self.levels)
		]

		# Define threshold parameters
		self.alpha = tensor(10.)
		self.b_plus = tensor(1.)
		self.b_minus = tensor(1.)

		# Define hard thresholding layer
		self.threshold = HardThreshold(self.alpha, self.b_plus, self.b_minus)

		# Define backward transformations
		self.backward_transforms = [
		    BackwardTransformLayer(self.wavelet[level, :])
		    for level in reversed(range(self.levels))
		]

	def forward(self, input):
		# Pad the input on the second axis to the length of a power of two.
		input_length = input.shape[1]

		desired_input_length = 2**(ceil(log2(input_length)))

		# Prepares a padder if the input size is not a power of two.
		padder = nn.ReplicationPad1d((0, desired_input_length - input_length))

		# Pad the input to an even size.
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

		return approximation[:, :input_length], reduce(
		    lambda tensor1, tensor2: concat(tensors=(tensor1, tensor2), dim=-1),
		    wav_coeffs)
