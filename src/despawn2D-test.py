from datetime import datetime

from pywt import Wavelet
from torch import device, manual_seed, save, tensor
from torch.nn import MSELoss as MainLoss
from torch.utils.data import DataLoader
from loader import MeasurementFolder

from train_model import *

# Set random seed for desterministic behaviour
manual_seed(0)

device = device("cuda:0")
EPOCHS = 3000
LEARNING_RATE = 1e-4
LR_HT = 3e-1
REG_FACT = 1e-1
REC_FACT = 1e-7
MEASUREMENT = "eps_yl_k3"
# More entries -> longer filters seem to be better. NN can adapt more I think.
WAVELET_NAME = "db20"
FILTER_LEN_FACTOR = 20
FILTER = True
LEVELS = 13

#signal, data = make_noisy_sine(device=device)

dataset = DataLoader(
    MeasurementFolder("store/"),
    pin_memory=True,
    batch_size=None,
)

# Sets the scaling / wavelet function. If it is repeated so we have more weights
# we also scale it down so the total result of the convolution does tripple.
SCALING = tensor(Wavelet(WAVELET_NAME).dec_hi,
                 device=device).repeat(FILTER_LEN_FACTOR) / FILTER_LEN_FACTOR

loss_fn = MainLoss(reduction="mean")

model = make_model(
    LEVELS,
    SCALING.to(device),
).to(device)

optimizer = make_optimizer(model, LEARNING_RATE, LR_HT)

train_loop(model, optimizer, dataset, EPOCHS, WAVELET_NAME)

save(
    model,
    f"models/despawn-reg-{EPOCHS}-meas-{MEASUREMENT}-wavelet-{WAVELET_NAME}.pt")
