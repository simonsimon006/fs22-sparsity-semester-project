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
EPOCHS = 50
LEARNING_RATE = 1e-2
LR_HT = 3e-1
REG_FACT = 1e-1
REC_FACT = 1e-7
MEASUREMENT = "eps_yl_k3"
# More entries -> longer filters seem to be better. NN can adapt more I think.
WAVELET_NAME = "db10"
FILTER_LEN_FACTOR = 1
FILTER = True
LEVELS = 10

#signal, data = make_noisy_sine(device=device)

dataset = DataLoader(
    MeasurementFolder("store/"),
    #pin_memory=True,
    batch_size=None,
)

# Sets the scaling / wavelet function. If it is repeated so we have more weights
# we also scale it down so the total result of the convolution does tripple.
SCALING = tensor(Wavelet(WAVELET_NAME).dec_hi,
                 device=device).repeat(FILTER_LEN_FACTOR) / FILTER_LEN_FACTOR

FILTER = map(tensor, Wavelet(WAVELET_NAME).filter_bank)

FILTER = map(lambda co: co.repeat(FILTER_LEN_FACTOR).cuda() / FILTER_LEN_FACTOR,
             FILTER)
loss_fn = MainLoss(reduction="mean")

model = make_model(
    LEVELS,
    SCALING,
).to(device)

optimizer = make_optimizer(model, LEARNING_RATE, LR_HT)

train_loop(model, optimizer, wavelet_name="db2", epochs=EPOCHS)

save(
    model,
    f"models/despawn-reg-{EPOCHS}-meas-{MEASUREMENT}-wavelet-{WAVELET_NAME}.pt")

print(model.scaling)