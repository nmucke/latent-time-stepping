import pdb
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import yaml
from latent_time_stepping.datasets.AE_dataset import AEDataset

from latent_time_stepping.utils import load_trained_AE_model, load_trained_time_stepping_model
torch.set_default_dtype(torch.float32)


NUM_SKIP_STEPS = 5

DEVICE = 'cpu'

PHASE = "single"
AE_MODEL_TYPE = "WAE"
TIME_STEPPING_MODEL_TYPE = "transformer"

AE_model_path = f"trained_models/autoencoders/{PHASE}_phase_{AE_MODEL_TYPE}"

time_stepping_model_path = f"trained_models/time_steppers/{PHASE}_phase_{TIME_STEPPING_MODEL_TYPE}"


PREPROCESSOR_PATH = f'trained_preprocessors/{PHASE}_phase_preprocessor.pt'
preprocessor = torch.load(PREPROCESSOR_PATH, map_location=DEVICE)

BUCKET_NAME = "train_models"
ORACLE_SAVE_PATH = f'{PHASE}_phase/{AE_MODEL_TYPE}_{TIME_STEPPING_MODEL_TYPE}'
