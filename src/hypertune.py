import pathlib as pl
from datetime import datetime

import torch
from pywt import Wavelet
from ray import tune
from ray.air import RunConfig
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch as SearchAlgo
from ray.tune.utils import wait_for_gpu

from train_model import *

NOW = str(datetime.now().timestamp()).replace(".", "_")

run_config = RunConfig(name=NOW)
wavelist = {
    name: torch.tensor(Wavelet(name).dec_hi)
    for name in ["db20", "dmey", "db2", "db10", "bior6.8", "sym20", "sym10"]
}

factors = [5, 10, 20]
for fac in factors:
	wavelist[f"{fac}xdmey"] = (torch.tensor(Wavelet("dmey").dec_hi) /
	                           fac).repeat(fac)


def tune_loop(config):
	torch.cuda.empty_cache()
	wait_for_gpu(target_util=0.3)
	model = make_model(config["levels"], config["wav"][1])
	opt = make_optimizer(model, 1e-4, 3e-1)
	train_loop(model=model,
	           optimizer=opt,
	           wavelet_name=config["wav"][0],
	           epochs=10000)


config = {
    "levels": tune.randint(2, 17),
    "wav": tune.choice(wavelist.items()),
}
scheduler = ASHAScheduler(time_attr="training_iteration",
                          max_t=5000,
                          grace_period=5,
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
    num_samples=20,
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

torch.save(best_result, "best_model.pt")

print(best_config)
best_result.metrics_dataframe.to_json("best_result_metrics.json")
