from functools import reduce
from math import ceil, log2

from pywt import Wavelet
from torch import Tensor, concat, conj, flip, nn, tensor, zeros, device
from torch.fft import irfft, rfft
from torch.nn import ReplicationPad1d
from torch.nn.functional import sigmoid
from torch.nn.parameter import Parameter


def compute_wavelet(scaling_rec: Tensor) -> Tensor:
	'''Computes the wavelet filter coefficients from the scaling coefficients. The formula is g_k = h_(M-k-1) where M is the filter length of h. The formula stems from a paper.'''

	# flip copies / clones the input.
	g = flip(scaling_rec, (-1, ))
	g[1::2] *= -1

	return g


def compute_wavelet_synthesis(scaling: Tensor) -> Tensor:
	# flip copies / clones the input.
	g = flip(scaling, (-1, ))
	g[1::2] *= -1

	return g


def convolve(input, filter, transpose=False):
	'''This function is my attempt at using the FFT to do the convolution. It is much faster than the normal convolution function from PyTorch.'''

	# Length is the length of the input. Some problems with that.
	input_length = input.shape[-1]

	ra = rfft(input)
	if transpose:
		rb = conj(rfft(filter, n=input_length))
	else:
		rb = rfft(filter, n=input_length)

	res = irfft(input=ra * rb, n=input_length)

	return res


class ForwardTransformLayer(nn.Module):
	'''Expects a tensor with shape (n, m) and computes one forward step over each row, i.e. it iterates over all values of 0...n and then computes the lifting. input_length needs to be equal to m.'''

	def __init__(self, scaling, scaling_rec):
		super(ForwardTransformLayer, self).__init__()

		# Store the scaling function filter.
		self.scaling = scaling
		self.scaling_rec = scaling_rec

	def forward(self, input):
		wavelet = compute_wavelet(self.scaling_rec)

		details = convolve(input, wavelet)
		approximation = convolve(input, self.scaling)

		return details[:, ::2], approximation[:, ::2]


class BackwardTransformLayer(nn.Module):
	'''Expects a tensor with shape (n, m) and computes one backward step over each row, i.e. it iterates over all values of 0...n and then computes the lifting. input_length needs to be equal to m.'''

	def __init__(self, scaling, scaling_rec):
		super(BackwardTransformLayer, self).__init__()

		# Store the filters.
		self.scaling = scaling
		self.scaling_rec = scaling_rec

		self.transpose = True

	def forward(self, input):

		# Compute synthesis filter coefficients
		wavelet_synthesis = compute_wavelet_synthesis(self.scaling_rec)
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

		# Convolve with the reconstruction filters.
		details_rec = convolve(filled_details,
		                       wavelet_synthesis,
		                       transpose=self.transpose)
		approx_rec = convolve(filled_approximation,
		                      scaling_synthesis,
		                      transpose=self.transpose)

		return details_rec + approx_rec


class HardThreshold(nn.Module):

	def __init__(self, alpha=10, b_plus=2, b_minus=1):
		super(HardThreshold, self).__init__()
		self.alpha = Parameter(alpha)
		self.b_plus = Parameter(b_plus)
		self.b_minus = Parameter(b_minus)

	def forward(self, input):
		return input * (sigmoid(self.alpha * (input - self.b_plus)) +
		                sigmoid(-self.alpha * (input + self.b_minus)))


class Despawn2D(nn.Module):
	'''Expects the data to come in blocks with shape (n, m). The blocks can have different shapes but the rows of the blocks have to have constant length within a block.'''

	def __init__(self,
	             levels: int,
	             wavelet_name: str,
	             adapt_filters: bool = True,
	             device: str | device = "cpu"):
		super(Despawn2D, self).__init__()

		# Extract filter coefficients from the wavelet object.
		scaling = tensor(Wavelet(wavelet_name).dec_hi, device=device)
		scaling_rec = tensor(Wavelet(wavelet_name).rec_hi, device=device)

		self.scaling = Parameter(scaling.repeat(levels, 1), requires_grad=True)
		self.scaling_rec = Parameter(scaling_rec.repeat(levels, 1),
		                             requires_grad=adapt_filters)

		self.levels = levels

		# Define forward transforms
		self.forward_transforms = [
		    ForwardTransformLayer(self.scaling[level, :],
		                          self.scaling_rec[level, :])
		    for level in range(self.levels)
		]

		# Define threshold parameters similar to the paper
		self.alpha = tensor(10.)
		self.b_plus = tensor(2.)
		self.b_minus = tensor(1.)

		# Define hard thresholding layer
		self.threshold = HardThreshold(self.alpha, self.b_plus, self.b_minus)

		# Define backward transformations
		self.backward_transforms = [
		    BackwardTransformLayer(self.scaling[level, :],
		                           self.scaling_rec[level, :])
		    for level in reversed(range(self.levels))
		]

	def forward(self, input):
		# Pad the input on the second axis to the length of a power of two.
		input_length = input.shape[1]

		desired_input_length = 2**(ceil(log2(input_length)))

		# Pads the input if the input size is not a power of two.
		# The circular padding is necessary for perfect reconstruction.
		# And it gives the least amount of coefficients.
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
		# The index stuff with approximation removes the padding. The reduc
		# call concatenates all wavelet coefficients into one big tensor.
		return approximation[:, to_pad[0]:input_length + to_pad[1]], reduce(
		    lambda tensor1, tensor2: concat(tensors=(tensor1, tensor2), dim=-1),
		    wav_coeffs)
