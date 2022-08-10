from typing import Tuple
from torch import tensor, Tensor
from util import convolve


def forward_step(input: Tensor, predict: Tensor,
                 update: Tensor) -> Tuple[Tensor, Tensor]:
	'''The function performs one lifting step.'''
	even = input[:, ::2]
	odd = input[:, 1::2]

	odd_prediction = convolve(even, predict)

	difference = odd - odd_prediction

	update_even = convolve(difference, update)

	updated_even = even - update_even

	return updated_even, difference


def backward_step(updated_even: Tensor, difference: Tensor, predict: Tensor,
                  update: Tensor) -> Tensor:
	update_even = convole(difference, update)
	even = updated_even + update_even

	odd_prediction = convolve(even, predict)
	odd = difference + odd_prediction

	result = tensor()