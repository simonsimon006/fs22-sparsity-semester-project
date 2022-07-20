import numpy as np
import torch as t
from matplotlib import pyplot as plt

from plotter import plot_measurement
from util import make_noisy_sine

signal, noisy, time = make_noisy_sine(spatial_resolution=30)

from pysr import PySRRegressor

model = PySRRegressor(
    model_selection="best",  # Result is mix of simplicity+accuracy
    niterations=40,
    binary_operators=["+", "*", "^", "-", "/"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",
        "sinh",
        "cosh",
        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    loss="loss(x, y) = (x - y)^2",
    # ^ Custom loss function (julia syntax),
    select_k_features=2,
    populations=15,
)

Y = noisy.numpy()
model.fit(time.numpy(), Y)
print(model.equations)
