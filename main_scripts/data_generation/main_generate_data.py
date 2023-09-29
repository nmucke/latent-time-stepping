import time
import numpy as np
from discontinuous_galerkin.base.base_model import BaseModel
from discontinuous_galerkin.start_up_routines.start_up_1D import StartUp1D
import matplotlib.pyplot as plt
import pdb
import oci
import scipy.stats
import yaml

from matplotlib.animation import FuncAnimation
import ray
from generate_data_utils import simulate_pipeflow
from single_phase_PDE import PipeflowEquations as PipeflowEquationsSinglePhase
from multi_phase_PDE import PipeflowEquations as PipeflowEquationsMultiPhase

TEST_CASE = 'multi_phase_leak'

DISTRIBUTED = True
NUM_CPUS = 10

NUM_SAMPLES = 10
TRAIN_OR_TEST = 'test'

TO_ORACLE = True

# Load .yml config file
with open(f'configs/PDEs/{TEST_CASE}.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
if TEST_CASE == 'single_phase_leak':
    model_parameters = {
        'L': 1000,
        'd': 0.508,
        'A': np.pi*0.508**2/4,
        'c': 308.,
        'rho_ref': 52.67,
        'p_amb': 101325.,
        'p_ref': 52.67*308**2,
        'e': 1e-8,
        'mu': 1.2e-5,
        'Cd': 5e-4,
        'leak_location': 500,
    }
    
elif TEST_CASE == 'multi_phase_leak':
    model_parameters = {
        'L': 5000, # meters
        'd': 0.2, # meters
        'A': np.pi*0.2**2/4, # meters^2
        'c': 308., # m/s
        'rho_g_norm': 1.26, # kg/m^3
        'rho_l': 1003., # kg/m^3
        'p_amb': 1.01325e5, # Pa
        'p_norm': 1.0e5, # Pa
        'p_outlet': 1.0e6, # Pa
        'e': 1e-8, # meters
        'mu_g': 1.8e-5, # Pa*s
        'mu_l': 1.516e-3, # Pa*s
        'T_norm': 278, # Kelvin
        'T': 278, # Kelvin
        'Cd': 0.1,
        'leak_location': 500,
    }

def main():

    if TEST_CASE == 'single_phase_leak':
        pipe_equations = PipeflowEquationsSinglePhase
    elif TEST_CASE == 'multi_phase_leak':
        pipe_equations = PipeflowEquationsMultiPhase

    if DISTRIBUTED:
        ray.shutdown()
        ray.init(num_cpus=NUM_CPUS)
        simulate_pipeflow_remote = ray.remote(simulate_pipeflow)
        remote_list = []


    if TEST_CASE == 'single_phase_leak':
        leak_location_vec = np.random.uniform(10, 990, NUM_SAMPLES)
        leak_size_vec = np.random.uniform(1.0, 3.0, NUM_SAMPLES)
    elif TEST_CASE == 'multi_phase_leak':
        leak_location_vec = np.random.uniform(10, 4990, NUM_SAMPLES)
        leak_size_vec = np.random.uniform(1.0, 2.0, NUM_SAMPLES)



    for idx in range(NUM_SAMPLES):

        if DISTRIBUTED:
            remote_list.append(simulate_pipeflow_remote.remote(
                PDE_model=pipe_equations,
                PDE_args=config,
                t_final=180.0,
                model_parameters=model_parameters,
                parameters_of_interest={
                    'leak_size': leak_size_vec[idx],
                    'leak_location': leak_location_vec[idx],
                },
                phase='single' if TEST_CASE == 'single_phase_leak' else 'multi',
                idx=idx,
                train_or_test=TRAIN_OR_TEST,
                to_oracle=TO_ORACLE,
                )
            )


        else:
            _ = simulate_pipeflow(
                PDE_model=pipe_equations,
                PDE_args=config,
                t_final=180.0,
                model_parameters=model_parameters,
                parameters_of_interest={
                    'leak_size': leak_size_vec[idx],
                    'leak_location': leak_location_vec[idx],
                },
                phase='single' if TEST_CASE == 'single_phase_leak' else 'multi',
                idx=idx,
                train_or_test=TRAIN_OR_TEST,
                to_oracle=TO_ORACLE,
            )
    
    ray.get(remote_list)

    if DISTRIBUTED:
        ray.shutdown()


if __name__ == "__main__":
    
    main()
