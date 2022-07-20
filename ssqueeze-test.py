import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt, ssq_stft, icwt
import os
from torch import load, device, float32
from src.plotter import plot_measurement

os.environ['SSQ_GPU'] = '0'
device = device("cuda:0")
MEASUREMENT = "eps_yl_k3"
signal = load("signal.pt").to(dtype=float32).numpy()
data = load("data.pt").to(dtype=float32).numpy()

a = ssq_cwt(data)
res = icwt(a[0])

plot_measurement(signal, data, res, "ssq_cwt-test.png", MEASUREMENT)