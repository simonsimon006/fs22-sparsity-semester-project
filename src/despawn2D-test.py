from re import S
from despawn2D import Despawn2D

from torch.optim import Adadelta as Opt
from torch.nn import MSELoss as MainLoss, L1Loss
from torch import tensor, norm, save, sin, linspace, pi, manual_seed, zeros_like, device, rand_like, set_default_dtype, float64
import matplotlib.pyplot as plt

# Set random seed for desterministic behaviour
manual_seed(0)
#set_default_dtype(float64)
device = device("cuda:0")

TESTING_FREQ = 5 * 2 * pi  # Hz
AMPLITUDE = 10

EPOCHS = 1000
LEARNING_RATE = 1

# db2, random choice
SCALING = tensor([
    -0.12940952255092145,
    0.22414386804185735,
    0.836516303737469,
    0.48296291314469025,
],
                 device=device)

REG_FACT = 1e-6

time = linspace(0, 5, 4096, device=device)
signal = AMPLITUDE * sin(TESTING_FREQ * time).repeat(500, 1)
noise = rand_like(signal, device=device)
#noise = zeros_like(signal, device=device)
data = (signal + noise)

loss_fn = MainLoss(reduction="sum")
#regularization = L1Loss(reduction="sum")
model = Despawn2D(10, SCALING)
model = model.to(device)

optimizer = Opt(model.parameters(), lr=LEARNING_RATE)
'''for param in model.parameters():
	print(param)
	print(param.device)
'''
for epoch in range(EPOCHS):
	optimizer.zero_grad()

	denoised, wav_coeffs = model(data)

	# Do regularization
	reg_loss = norm(wav_coeffs, p=1)
	error = loss_fn(signal, denoised)

	loss = error + REG_FACT * reg_loss
	loss.backward()

	optimizer.step()

	if epoch % 2 == 0:
		print(
		    f"Epoch {epoch+1} of {EPOCHS}: Reconstruction error: {float(error)}; Regularization penalty: {float(reg_loss)}; total loss: {float(loss)}"
		)

print(list(model.parameters()))

plt.plot(time.cpu(), denoised.cpu().detach().numpy()[0, :], label="Denoised")

plt.plot(time.cpu(),
         signal.cpu().detach().numpy()[0, :],
         label="Signal",
         alpha=0.2)
plt.legend()
plt.show()
save(model, f"despawn2d-{EPOCHS}-sine-{TESTING_FREQ}_Hz.pt")