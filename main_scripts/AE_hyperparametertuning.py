import pdb
from matplotlib import pyplot as plt
import numpy as np
import ray
import yaml
from yaml.loader import SafeLoader
import torch

from ray import tune
from ray.air import session
from ray.tune.search.hyperopt import HyperOptSearch

from latent_time_stepping.AE_models.VAE_encoder import VAEEncoder

from latent_time_stepping.AE_models.autoencoder import Autoencoder

from latent_time_stepping.AE_models.encoder_decoder import (
    Decoder, 
    Encoder
)
from latent_time_stepping.datasets.AE_dataset import get_AE_dataloader
from latent_time_stepping.AE_training.optimizers import Optimizer
from latent_time_stepping.AE_training.train_steppers import WAETrainStepper
from latent_time_stepping.AE_training.trainer import train

torch.set_default_dtype(torch.float32)

config_path = f"configs/WAE.yml"
with open(config_path) as f:
    config = yaml.load(f, Loader=SafeLoader)

def fitness_function(
    config,
    train_data,
    val_data,
    ):

    CUDA = True
    if CUDA:
        DEVICE = torch.device('cuda' if CUDA else 'cpu')
    else:
        DEVICE = torch.device('cpu')

    DEVICE = torch.device('cpu')

    encoder = Encoder(**config['model_args']['encoder'])
    decoder = Decoder(**config['model_args']['decoder'])

    model = Autoencoder(
        encoder=encoder,
        decoder=decoder,
    )
    model = model.to(DEVICE)

    optimizer = Optimizer(
        model=model,
        args=config['optimizer_args'],
    )

    train_stepper = WAETrainStepper(
        model=model,
        optimizer=optimizer,
        **config['train_stepper_args'],
    )

    train_dataloader = get_AE_dataloader(
        state=train_data['state'],
        pars=train_data['pars'],
        **config['dataloader_args']
    )
    val_dataloader = get_AE_dataloader(
        state=val_data['state'],
        pars=val_data['pars'],
        **config['dataloader_args']
    )
    
    val_metric = train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model_save_path=None,
        train_stepper=train_stepper,
        print_progress=True,
        return_val_metrics=True,
        **config['train_args'],
    )
    
    session.report({"config": config})
    session.report({"loss": val_metric})

    
def main():

    STATE_PATH = 'data/processed_data/training_data/states.pt'
    PARS_PATH = 'data/processed_data/training_data/pars.pt'

    TRAIN_SAMPLE_IDS = range(50)
    VAL_SAMPLE_IDS = range(50, 60)

    state = torch.load(STATE_PATH)
    pars = torch.load(PARS_PATH)

    train_state = state[TRAIN_SAMPLE_IDS]
    train_pars = pars[TRAIN_SAMPLE_IDS]
    train_data = {
        'state': train_state,
        'pars': train_pars,
    }

    val_state = state[VAL_SAMPLE_IDS]
    val_pars = pars[VAL_SAMPLE_IDS]
    val_data = {
        'state': val_state,
        'pars': val_pars,
    }
    
    config = {
        "l1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16]),
    }
    current_best_config = [{
        "l1": 2**np.random.randint(2, 9),
        "l2": 2**np.random.randint(2, 9),
        "lr": 1e-4,
        "batch_size": 2,
    }]

    algo = HyperOptSearch(
        points_to_evaluate=current_best_config,
        metric="mean_loss", 
        mode="min"
        )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(lambda config: fitness_function(config, train_data, val_data)),
            resources={"cpu": 2, "gpu": 0}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            search_alg=algo,
            num_samples=10,
            ),
        param_space=config
        )

    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)


if __name__ == "__main__":
    
    ray.shutdown()
    ray.init(num_cpus=10)
    main()
    ray.shutdown()
