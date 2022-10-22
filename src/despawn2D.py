from functools import reduce

from torch import Tensor, nn, tensor, sigmoid, squeeze, device, has_cuda, float32
from torch.nn.parameter import Parameter

from despawn1D import (BackwardTransformLayer, ForwardTransformLayer,
                       HardThreshold)
from util import PadState, compute_scaling_synthesis, compute_wavelet, compute_wavelet_synthesis, l1_reg, pad_wrap
import ptwt
from pywt import Wavelet
from ptwt.wavelets_learnable import SoftOrthogonalWavelet

PADD = 50


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
	             padding=PADD,
	             wavelet_name="db20"):
		super(Despawn2D, self).__init__()
		# Allow for random initialization of the filters.
		# The levels * 2 is so we can use different filters for the row and
		# column decomposition.

		self.bank = SoftOrthogonalWavelet(*(scaling))
		self.bank.training = True
		self.device = "cpu"  #"cuda:0" if has_cuda else "cpu"

		self.levels = levels
		self.filter = filter
		self.wavelet_name = wavelet_name

		# Define hard thresholding layer
		self.alpha = Parameter(tensor(10.))
		self.b_plus = Parameter(tensor(2.))
		self.b_minus = Parameter(tensor(1.))

		self.threshold = HardThreshold(self.alpha, self.b_plus, self.b_minus)
		self.padding = padding
		self.scaling = Parameter(scaling.repeat(levels * 2, 1).to(self.device),
		                         requires_grad=adapt_filters)
		# Define forward transforms
		self.forward_transforms = [
		    ForwardTransform2D(self.scaling[level, :],
		                       self.scaling[level + 1, :])
		    for level in range(self.levels)
		]

		# Define backward transformations
		self.backward_transforms = [
		    BackwardTransform2D(self.scaling[level, :],
		                        self.scaling[level + 1, :])
		    for level in reversed(range(self.levels))
		]

	def forward(self, input):
		# I hope to cut-off boundary errors this way.
		input = pad_wrap(
		    input, (self.padding, self.padding, self.padding, self.padding),
		    mode="replicate")
		approximation = input.to(self.device)
		# Store the info about when we pad levels. We pad the different
		# levels to an even size, so that we do not need all the coefficients
		# for an overall padding.
		# Also, this way we can use arbitrary numbers of levels and do not need
		# to stick to the max size of the normal decomposition. This is useful
		# if, e.g. the vertical dimension is much larger than the horizontal
		# one.
		pad_info = []
		wav_coeffs = []
		for transform in self.forward_transforms:
			# Prepare our markers for padding.
			vert = False
			horiz = False

			# Check how we need to pad the input
			if approximation.shape[0] % 2 != 0:
				v_pad = (0, 1)
				vert = True
			else:
				v_pad = (0, 0)
			if approximation.shape[1] % 2 != 0:
				h_pad = (0, 1)
				horiz = True
			else:
				h_pad = (0, 0)

			# Create and enter a new padding entry.
			padinfo = PadState(vert, horiz)
			pad_info.append(padinfo)

			# Create the structure that tells torchs pad function
			# how to pad our input.
			to_pad = (*h_pad, *v_pad)

			# Pad the input with our convenience wrapper.
			approximation = pad_wrap(approximation, to_pad)

			details, approximation = transform(approximation)

			wav_coeffs.append(details)
		wav_coeffs.append(approximation)

		# Filter the coefficients if the global setting is enabled.
		#filtered = []
		if self.filter:
			filtered = list(map(self.threshold, wav_coeffs))
			'''for (hh, hl, lh) in wav_coeffs[:-1]:
				bh = hh
				hh = self.threshold_hh(hh)
				hl = self.threshold_hl(hl)
				lh = self.threshold_lh(lh)
				filtered.append((hh, hl, lh))'''
			#filtered.append(self.threshold(approximation))
		else:
			filtered = list(wav_coeffs)

		# Computes the regularisation loss on the filtered wavelet coefficients.
		# level[0] is the level, level[1] are the actual coeffs.
		# The idea is to penalize finer details.

		#reg_loss = reduce(lambda acc, level: acc + self.reg_loss_fun(level),
		#                  filtered, 0)
		#reg_loss /= len(filtered)

		#reg_loss = self.reg_loss_fun(wav_coeffs)
		# Transform the filtered inputs back.
		res_filtered = list(filtered)
		approximation = filtered.pop()

		for transform in self.backward_transforms:
			level_pad_info = pad_info.pop()
			details = filtered.pop()
			approximation = transform((*details, approximation))

			# If we padded the level we need to cut it back again.
			if level_pad_info.vert:
				approximation = approximation[:-1, :]
			if level_pad_info.horiz:
				approximation = approximation[:, :-1]

		return approximation[self.padding:-self.padding,
		                     self.padding:-self.padding], res_filtered
