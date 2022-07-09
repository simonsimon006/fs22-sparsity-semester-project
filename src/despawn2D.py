from functools import reduce
from math import ceil, floor, log2

from torch import Tensor, nn, rand, squeeze, tensor, unsqueeze
from torch.nn.functional import pad
from torch.nn.parameter import Parameter

from despawn1D import (BackwardTransformLayer, ForwardTransformLayer,
                       HardThreshold, l1_reg)


class ForwardTransform2D(nn.Module):
	'''Implements the 2D forward wavelet transform using the previously 1D 
	transform.'''

	def __init__(self, scaling_H, scaling_V):
		super(ForwardTransform2D, self).__init__()
		self.forward_H = ForwardTransformLayer(scaling_H)
		self.forward_V = ForwardTransformLayer(scaling_V)

	def forward(self, input):
		# First we transform horizontally
		h_detail, h_approx = self.forward_H(input)
		# We concat the details and approximations from above as
		# tensor((details, approx)) and transpose them so we can process them
		# vertically. Then we concat that output again and transpose it back.

		hh, hl = self.forward_V(h_detail.T)
		hh, hl = hh.T, hl.T

		lh, ll = self.forward_V(h_approx.T)
		lh, ll = lh.T, ll.T

		return (hh, hl, lh), ll


class BackwardTransform2D(nn.Module):
	'''Implements a 2D wavelet transform using the previously implemented 1D case.'''

	def __init__(self, scaling_H, scaling_V):
		super(BackwardTransform2D, self).__init__()

		self.backward_H = BackwardTransformLayer(scaling_H)
		self.backward_V = BackwardTransformLayer(scaling_V)

	def forward(self, input):
		# Build the whole tensor to implement the transforms. The input is expected to have the order (hh, hl, lh, ll)
		(hh, hl, lh, ll) = input

		# First reverse the columns.
		h = self.backward_V((hh.T, hl.T)).T
		l = self.backward_V((lh.T, ll.T)).T

		# Then reverse the rows.
		inverse = self.backward_H((h, l))

		return inverse


class Despawn2D(nn.Module):
	'''Expects the data to come in blocks with shape (n, m). The blocks can have different shapes but the rows of the blocks have to have constant length within a block.'''

	def __init__(self,
	             levels: int,
	             scaling: Tensor,
	             adapt_filters: bool = True,
	             filter=True,
	             reg_loss_fun=l1_reg):
		super(Despawn2D, self).__init__()
		# Allow for random initialization of the filters.
		if type(scaling) == int:
			self.scaling = Parameter(rand(levels * 2, scaling),
			                         requires_grad=adapt_filters)
		else:
			self.scaling = Parameter(scaling.repeat(levels * 2, 1),
			                         requires_grad=adapt_filters)

		self.levels = levels

		# Define forward transforms
		self.forward_transforms = [
		    ForwardTransform2D(self.scaling[level, :],
		                       self.scaling[level + 1, :])
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
		    BackwardTransform2D(self.scaling[level, :],
		                        self.scaling[level + 1, :])
		    for level in reversed(range(self.levels))
		]
		self.filter = filter
		self.reg_loss_fun = reg_loss_fun

	def forward(self, input):
		# Pad the input on the second axis to the length of a power of two.
		n, m = input.shape

		n_level = ceil(log2(n))
		m_level = ceil(log2(m))

		desired_n = 2**(n_level)
		desired_m = 2**(m_level)

		# Pads the input if the input size is not a power of two.
		# The circular padding is necessary for perfect reconstruction.
		# And it gives the least amount of coefficients.

		# First the padding for n
		side_n_pad = (desired_n - n) / 2
		to_pad_n = (floor(side_n_pad), ceil(side_n_pad))

		# Then the padding for m
		side_m_pad = (desired_m - m) / 2
		to_pad_m = (floor(side_m_pad), ceil(side_m_pad))

		# Now the overall pad.
		to_pad = (*to_pad_m, *to_pad_n)

		input = unsqueeze(input, dim=0)
		input = unsqueeze(input, dim=0)

		input = pad(input, to_pad, mode="replicate")

		input = squeeze(input, dim=0)
		input = squeeze(input, dim=0)

		approximation = input
		wav_coeffs = []
		max_level = min(n_level, m_level) + 1
		for transform in self.forward_transforms[:max_level]:
			details, approximation = transform(approximation)
			wav_coeffs.append(details)
		wav_coeffs.append(approximation)

		# Filter the coefficients if the global setting is enabled.
		if self.filter:
			filtered = list(map(self.threshold, wav_coeffs))
		else:
			filtered = list(wav_coeffs)

		# Computes the regularisation loss on the filtered wavelet coefficients.
		reg_loss = reduce(lambda acc, coeffs: acc + self.reg_loss_fun(coeffs),
		                  filtered, 0)

		# Transform the filtered inputs back.
		approximation = filtered.pop()
		for transform in self.backward_transforms[:max_level]:
			details = filtered.pop()
			approximation = transform((*details, approximation))

		# The index stuff with approximation removes the padding.
		approx_cut = approximation[to_pad_n[0]:to_pad_n[0] + n,
		                           to_pad_m[0]:to_pad_m[0] + m]

		return approx_cut, reg_loss
