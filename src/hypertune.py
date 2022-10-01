import pathlib as pl
from datetime import datetime

import torch
from pywt import Wavelet
import ray
from ray import tune, air
from ray.air import RunConfig
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch as SearchAlgo
from ray.tune.utils import wait_for_gpu

from train_model import *

NOW = str(datetime.now().timestamp()).replace(".", "_")
EPOCHS = 20
SAMPLES = 20

ray.init()

run_config = RunConfig(
    name=NOW,
    local_dir="ray_tuner/",
    sync_config=tune.SyncConfig(syncer=None),
    checkpoint_config=air.CheckpointConfig(
        num_to_keep=5, checkpoint_score_attribute="total_loss"),
)
wavelist = {
    name: torch.tensor(Wavelet(name).filter_bank)
    for name in ["db20", "dmey", "db10", "bior6.8", "sym20", "sym10"]
}
'''
factors = [5, 10, 20]
for fac in factors:
	wavelist[f"{fac}xdmey"] = (torch.tensor(Wavelet("dmey").dec_hi) /
	                           fac).repeat(fac)
'''


def tune_loop(config):
	torch.cuda.empty_cache()
	wait_for_gpu(target_util=0.3)
	model = make_model(config["levels"], config["wav"][1])
	opt = make_optimizer(model, 1e-4 / len(config["wav"][1][1]))

	train_loop(model=model,
	           optimizer=opt,
	           wavelet_name=config["wav"][0],
	           epochs=EPOCHS)


config = {
    "levels": tune.randint(5, 17),
    "wav": tune.choice(wavelist.items()),
}
scheduler = ASHAScheduler(time_attr="training_iteration",
                          max_t=EPOCHS,
                          grace_period=int(EPOCHS * 0.2),
                          reduction_factor=2)
reporter = CLIReporter(parameter_columns=[
    "levels",
],
                       metric_columns=[
                           "total_loss", "reconstruction_mse", "l0_pnorm",
                           "regularization_penalty"
                       ])

algo = SearchAlgo(random_state_seed=0)

tune_config = tune.TuneConfig(
    metric="total_loss",
    mode="min",
    num_samples=SAMPLES,
    scheduler=scheduler,
    search_alg=algo,
)
tuner = tune.Tuner(tune.with_resources(tune_loop, resources={
    "gpu": 1,
}),
                   tune_config=tune_config,
                   param_space=config,
                   run_config=run_config)
results = tuner.fit()

best_result = results.get_best_result()
best_config = best_result.config

torch.save(best_result.checkpoint, "best_model_and_opt_config.pt")

print(best_config)
best_result.metrics_dataframe.to_json("best_result_metrics.json")

print("Best checkpoint bank")
print(best_result.checkpoint)