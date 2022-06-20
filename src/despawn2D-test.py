from re import S

import matplotlib.pyplot as plt
from pywt import Wavelet
from torch import (device, float64, linspace, manual_seed, norm, pi, rand_like,
                   save, set_default_dtype, sin, tensor, zeros_like)
from torch.nn import L1Loss
from torch.nn import MSELoss as MainLoss
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Adadelta as Opt
from torch.optim.lr_scheduler import ReduceLROnPlateau as RLP

from despawn2D import Despawn2D

# Set random seed for desterministic behaviour
manual_seed(0)
#set_default_dtype(float64)
device = device("cpu")

TESTING_FREQ = 5 * 2 * pi  # Hz * rad
AMPLITUDE = 10
ADAPT_FILTERS = False
EPOCHS = 1000
LEARNING_RATE = 2 * 1e1

# Random choice
WAVELET_NAME = "db8"

SCALING = tensor(Wavelet(WAVELET_NAME).dec_hi, device=device)
print(SCALING.shape)
REG_FACT = 0

time = linspace(0, 5, 120, device=device)
signal = AMPLITUDE * sin(TESTING_FREQ * time).repeat(250, 1)
noise = rand_like(signal, device=device)
#noise = zeros_like(signal, device=device)
data = (signal + noise)

loss_fn = MainLoss(reduction="sum")
#regularization = L1Loss(reduction="sum")
model = Despawn2D(2, SCALING, adapt_filters=ADAPT_FILTERS)
model = model.to(device)

optimizer = Opt(model.parameters(), lr=LEARNING_RATE)
scheduler = RLP(optimizer, 'min', patience=2, factor=0.5)

for epoch in range(EPOCHS):
	optimizer.zero_grad()

	denoised, wav_coeffs = model(data)

	# Do regularization
	reg_loss = norm(wav_coeffs, p=1)
	error = loss_fn(signal, denoised)

	loss = error + REG_FACT * reg_loss
	#clip_grad_norm_(model.parameters(), 1)
	loss.backward()

	scheduler.step(loss)

	if epoch % 2 == 0:
		print(
		    f"Epoch {epoch+1} of {EPOCHS}: Reconstruction error: {float(error)}; Regularization penalty: {float(reg_loss)}; total loss: {float(loss)}"
		)

plt.plot(time.cpu(), denoised.cpu().detach().numpy()[0, :], label="Denoised")

plt.plot(time.cpu(),
         signal.cpu().detach().numpy()[0, :],
         label="Signal",
         alpha=0.2)
plt.plot(time.cpu(),
         data.cpu().detach().numpy()[0, :],
         label="Noisy signal",
         alpha=0.5)
plt.legend()

plt.savefig(
    f"sine-figures/despawn2d-{EPOCHS}-sine-{TESTING_FREQ}_Hz-amp-{AMPLITUDE}-adapt_filters-{ADAPT_FILTERS}-wavelet-{WAVELET_NAME}.svg"
)

plt.show()
save(
    model,
    f"models/despawn2d-{EPOCHS}-sine-{TESTING_FREQ}_Hz-amp-{AMPLITUDE}-wavelet-{WAVELET_NAME}.pt"
)
