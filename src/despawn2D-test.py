from datetime import datetime

from pywt import Wavelet
from torch import device, manual_seed, save, tensor
from torch.optim import AdamW as Opt
from torch.nn import MSELoss as MainLoss
from torch.utils.data import DataLoader
from loader import MeasurementFolder

from train_model import *

from util import make_sines

from tqdm import trange
# Set random seed for desterministic behaviour
manual_seed(0)

device = device("cuda:0")
EPOCHS = 50
LEARNING_RATE = 1e0

WAVELET_NAME = "db10"

LEVELS = 10
'''
dataset = DataLoader(
    MeasurementFolder("store/"),
    #pin_memory=True,
    batch_size=None,
)'''

dataset, signal = make_sines(30, 3000, 1000)
validation, val_sig = make_sines(1, 2000, 800)
val = validation[0].to("cuda:0")
val_sig = val_sig.to("cuda:0")
# Sets the scaling / wavelet function. If it is repeated so we have more weights
# we also scale it down so the total result of the convolution does tripple.
SCALING = tensor(Wavelet(WAVELET_NAME).filter_bank, device=device)

loss_fn = MainLoss(reduction="mean")

model = make_model(
    LEVELS,
    SCALING,
).to(device)

optimizer = Opt([{
    'params': model.threshold.parameters(),
    "lr": 1e0,
}, {
    'params': model.bank.parameters(),
    'lr': LEARNING_RATE
}],
                lr=1)

REG_FACT = 1e0
REC_FACT = 1e0
ORTH_FACT = 1e3
VAL_REC_FACT = 1e0


def combine_loss(rec, reg):
	return rec * REC_FACT + REG_FACT * reg


with trange(EPOCHS) as t:
	for epoch in t:
		for meas in dataset:
			denoised, coeffs = model(meas)
			target = signal.to(denoised.device)

			rec_loss = loss_fn(denoised, target)
			reg_loss = regularisation_fun(coeffs)
			orth_loss = model.bank.wavelet_loss()  # add wavelet_loss

			total = combine_loss(rec_loss, reg_loss) + ORTH_FACT * orth_loss

			total.backward()

			optimizer.step()
			optimizer.zero_grad()

		with inference_mode():
			denoised, coeffs = model(val)
			target = val_sig

			val_rec_loss = loss_fn(denoised, target)
			val_reg_loss = regularisation_fun(coeffs)
			val_l0_count = l0_counter(coeffs)

			val_total = VAL_REC_FACT * val_rec_loss + val_l0_count

		t.set_postfix(val_loss=float(val_total),
		              loss=float(total),
		              orth=float(orth_loss))
'''
save(
    model,
    f"models/despawn-reg-{EPOCHS}-meas-{MEASUREMENT}-wavelet-{WAVELET_NAME}.pt")

print(model.scaling)
'''