from functools import reduce

from torch import Tensor, tensor, zeros, sigmoid, float32
from torch.nn import Module

from torch.nn.parameter import Parameter
from torch.nn import ReflectionPad1d as Pad1d
from math import ceil, log2
from util import l1_reg, convolve, compute_wavelet


class ForwardTransformLayer(Module):
	'''Expects a tensor with shape (n, m) and computes one forward step over each row, i.e. it iterates over all values of 0...n and then computes the lifting. input_length needs to be equal to m.'''

	def __init__(self, scaling):
		super(ForwardTransformLayer, self).__init__()

		# Store the scaling function filter.
		self.scaling = scaling

	def forward(self, input):

		wavelet = compute_wavelet(self.scaling)

		details = convolve(input, wavelet)
		approximation = convolve(input, self.scaling)

		return details[:, ::2], approximation[:, ::2]


class BackwardTransformLayer(Module):
	'''Expects a tensor with shape (n, m) and computes one backward step over each row, i.e. it iterates over all values of 0...n and then computes the lifting. input_length needs to be equal to m.'''

	def __init__(self, scaling):
		super(BackwardTransformLayer, self).__init__()

		# Store the filters.
		self.scaling = scaling

	def forward(self, input):

		# Compute synthesis filter coefficients
		wavelet_synthesis = compute_wavelet(self.scaling)
		scaling_synthesis = self.scaling

		# Pad the input to an even size.
		(details, approximation) = input

		# We are creating new tensors so we have to tell them how they should look
		device = details.device
		# Interpolate with zeros.
		filled_details = zeros(details.shape[0],
		                       details.shape[1] * 2,
		                       device=device)
		filled_details[:, ::2] = details

		filled_approximation = zeros(approximation.shape[0],
		                             approximation.shape[1] * 2,
		                             device=device)

		filled_approximation[:, ::2] = approximation

		# Convolve with the reconstruction filters.
		details_rec = convolve(filled_details,
		                       wavelet_synthesis,
		                       transpose=True)
		approx_rec = convolve(filled_approximation,
		                      scaling_synthesis,
		                      transpose=True)

		return details_rec + approx_rec


class HardThreshold(Module):

	def __init__(self, alpha=10, b_plus=2, b_minus=1):
		super(HardThreshold, self).__init__()
		self.alpha = Parameter(tensor(alpha, dtype=float32))
		self.b_plus = Parameter(tensor(b_plus, dtype=float32))
		self.b_minus = Parameter(tensor(b_minus, dtype=float32))

	def forward(self, input):
		# The recursive application is to keep the structure of the
		# (hh, hl, lh) coeff tuples in an easy way.
		if type(input) == tuple:
			return [self.forward(elem) for elem in input]
		return input * (sigmoid(self.alpha * (input - self.b_plus)) +
		                sigmoid(-self.alpha * (input + self.b_minus)))


class Despawn(Module):
	'''Expects the data to come in blocks with shape (n, m). The blocks can have different shapes but the rows of the blocks have to have constant length within a block.'''

	def __init__(self,
	             levels: int,
	             scaling: Tensor,
	             adapt_filters: bool = True,
	             reg_loss_fun=l1_reg,
	             filter=True):
		super(Despawn, self).__init__()
		self.scaling = Parameter(scaling.repeat(levels, 1),
		                         requires_grad=adapt_filters)

		self.levels = levels

		# Define forward transforms
		self.forward_transforms = [
		    ForwardTransformLayer(self.scaling[level, :])
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
		    BackwardTransformLayer(self.scaling[level, :])
		    for level in reversed(range(self.levels))
		]
		self.reg_loss_fn = reg_loss_fun

	def forward(self, input):
		# Pad the input on the second axis to the length of a power of two.
		input_length = input.shape[1]

		desired_input_length = 2**(ceil(log2(input_length)))

		# Pads the input if the input size is not a power of two.
		# The circular padding is necessary for perfect reconstruction.
		# And it gives the least amount of coefficients.
		to_pad = ((desired_input_length - input_length) // 2,
		          ceil((desired_input_length - input_length) / 2))
		padder = Pad1d(to_pad)

		input = padder(input)

		approximation = input
		wav_coeffs = []
		for transform in self.forward_transforms:
			details, approximation = transform(approximation)
			wav_coeffs.append(details)
		wav_coeffs.append(approximation)

		# Filter the coefficients.
		filtered = list(map(self.threshold, wav_coeffs))
		#filtered = list(wav_coeffs)

		reg_loss = reduce(lambda acc, elem: acc + self.reg_loss_fn(elem),
		                  filtered, 0)

		# Transform the filtered inputs back.
		approximation = filtered.pop()
		for transform in self.backward_transforms:
			details = filtered.pop()
			approximation = transform((details, approximation))
		# The index stuff with approximation removes the padding. The reduc
		# call concatenates all wavelet coefficients into one big tensor.
		return approximation[:, to_pad[0]:input_length + to_pad[1]], reg_loss
