from torch import nn
from torch.fft import rfft, irfft
from torch import log2, ceil, flip, zeros


def even_odd(input):
	'''Splits a sequence into even and odd components.'''
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
	return irfft(ninput=rfft(a, norm=norm, n=length) * b, n=length, norm=norm)


class ForwardTransformLayer(nn.Module):
	'''Expects a tensor with shape (n, m) and computes one forward step over each row, i.e. it iterates over all values of 0...n and then computes the lifting. input_length needs to be equal to m.'''

	def __init__(self, input_length, wavelet, scaling):
		super(ForwardTransformLayer, self).__init__()

		# Set normalization mode for rfft and irfft.
		self.norm = "ortho"

		# Sets input length to the next larger power of two.
		'''self.padded_input_length = input_length if input_length % 2 == 0 else input_length + 1'''

		self.padded_input_length = 2**(ceil(log2(input_length)))
		# Set the length of the even and odd parts.
		self.part_length = self.padded_input_length // 2

		# Store the wavelet and scaling function filters.
		self.wavelet = wavelet
		self.scaling = scaling

		# Prepares a padder if the input size is not a power of two.
		self.padder = nn.ReflectionPad1d(
		    (0, self.padded_input_length - input_length
		     )) if self.padded_input_length != input_length else nn.Identity()

	def forward(self, input):
		# Pad the input to an even size.
		input = self.padder(input)

		# Splits the filter coefficients into even and odd parts and
		# transforms the wavelet and scaling coefficients.
		# Zero-padding happens in prepare_coefficients.

		wavelet_even, wavelet_odd = prepare(self.wavelet,
		                                    self.part_length,
		                                    norm=self.norm)
		scaling_even, scaling_odd = prepare(self.scaling,
		                                    self.part_length,
		                                    norm=self.norm)

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


class BackwardTransformLayer(nn.Module):
	'''Expects a tensor with shape (n, m) and computes one backward step over each row, i.e. it iterates over all values of 0...n and then computes the lifting. input_length needs to be equal to m.'''

	def __init__(self, wavelet, scaling):
		super(BackwardTransformLayer, self).__init__()

		# Set normalization mode for rfft and irfft.
		self.norm = "ortho"

		# Assume QMF.
		self.wavelet_rec = flip(wavelet, (-1, ))
		self.scaling_rec = flip(scaling, (-1, ))

	def forward(self, input):
		# Pad the input to an even size.
		(details, approximation) = input
		filled_details = zeros(details.shape[0], details.shape[1] * 2)
		filled_approximation = zeros(approximation.shape[0],
		                             approximation.shape[1] * 2)
		filled_details[:, ::2] = details
		filled_approximation[:, ::2] = approximation

		fil_details_len = filled_details.shape[1]
		fil_approx_len = filled_approximation.shape[1]

		rfft_wav_rec_padded = rfft(self.wavelet_rec,
		                           n=fil_details_len,
		                           norm=self.norm)
		rfft_scal_rec_padded = rfft(self.scaling_rec,
		                            n=fil_approx_len,
		                            norm=self.norm)

		# Convolve with the reconstruction filters.
		details_rec = convolve(filled_details, rfft_wav_rec_padded, self.norm,
		                       fil_details_len)
		approx_rec = convolve(filled_approximation, rfft_scal_rec_padded,
		                      self.norm, fil_approx_len)

		return details_rec + approx_rec
