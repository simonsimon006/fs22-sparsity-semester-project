import numpy as np
from scipy.io import loadmat

ds = loadmat("/home/simon/Code/quelltext/datasets/eps_yl_w3.mat")

from denoiser import denoise

X, E, Ec, t = denoise(ds["E_filled"])

from plotter import plot

plot(ds["E_filled"], X, E, Ec, t, "test.png")
