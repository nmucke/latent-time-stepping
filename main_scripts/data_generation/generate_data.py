import time
import numpy as np
from discontinuous_galerkin.base.base_model import BaseModel
from discontinuous_galerkin.start_up_routines.start_up_1D import StartUp1D
import matplotlib.pyplot as plt
import pdb
import scipy.stats
import yaml

from matplotlib.animation import FuncAnimation
import ray
from generate_data_utils import simulate_pipeflow
from single_phase_PDE import PipeflowEquations

TEST_CASE = 'single_phase_leak'

# Load .yml config file
with open(f'configs/PDEs/{TEST_CASE}.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    

if TEST_CASE == 'single_phase_leak':
    model_parameters = {
        'L': 2000,
        'd': 0.508,
        'A': np.pi*0.508**2/4,
        'c': 308.,
        'rho_ref': 52.67,
        'p_amb': 101325.,
        'p_ref': 52.67*308**2,
        'e': 1e-2,
        'mu': 1.2e-5,
        'Cd': 5e-4,
        'leak_location': 500,
    }


def main():
    simulate_pipeflow(
        PDE_model=PipeflowEquations,
        PDE_args=config,
        model_parameters=model_parameters,
        parameters_of_interest={
            'leak_size': 1.0,
            'leak_location': 500,
        },
        phase='single',
        idx=0
    )
    pdb.set_trace()

        
    ray.shutdown()
    ray.get([
        simulate_pipeflow.remote(
            leak_location=leak_location,
            leak_size=leak_size,
            pipe_DG=pipe_DG,
            idx=idx
        ) for (leak_location, leak_size, idx) in 
        zip(leak_location_vec, leak_size_vec, range(3000, 3000+num_samples))])

if __name__ == "__main__":
    
    #ray.shutdown()
    #ray.init(num_cpus=25)
    main()
    #ray.shutdown()
