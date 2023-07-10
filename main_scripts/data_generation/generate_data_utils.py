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

from single_phase_PDE import PipeflowEquations

def sol_to_state_single_phase(sol, t_vec, PDE_model):
    x = np.linspace(PDE_model.DG_vars.x[0, 0], PDE_model.DG_vars.x[-1, -1], 256)

    u = np.zeros((len(x), len(t_vec)))
    rho = np.zeros((len(x), len(t_vec)))
    for t in range(sol.shape[-1]):
        rho[:, t] = PDE_model.evaluate_solution(x, sol_nodal=sol[0, :, t])
        u[:, t] = PDE_model.evaluate_solution(x, sol_nodal=sol[1, :, t])
    rho = rho / PDE_model.A
    u = u / rho / PDE_model.A

    rho = rho[:, 1:]
    u = u[:, 1:]

    rho = np.expand_dims(rho, axis=0)
    u = np.expand_dims(u, axis=0)
    state = np.concatenate((rho, u), axis=0)

    return state

def pars_dict_to_array_single_phase(pars_dict):
    pars = np.zeros((2,))

    pars[0] = pars_dict['leak_location']
    pars[1] = pars_dict['leak_size']    

    return pars

def sol_to_state_multi_phase(sol, t_vec, PDE_model):

    x = np.linspace(PDE_model.DG_vars.x[0, 0], PDE_model.DG_vars.x[-1, -1], 512)

    A_l = np.zeros((len(x), len(t_vec)))
    p = np.zeros((len(x), len(t_vec)))
    u_m = np.zeros((len(x), len(t_vec)))
    for t in range(sol.shape[-1]):
        A_l[:, t] = PDE_model.evaluate_solution(x, sol_nodal=sol[0, :, t])
        p[:, t] = PDE_model.evaluate_solution(x, sol_nodal=sol[1, :, t])
        u_m[:, t] = PDE_model.evaluate_solution(x, sol_nodal=sol[2, :, t])
    
    alpha_l = A_l / PDE_model.A

    alpha_l = alpha_l[:, 1:]
    p = p[:, 1:]
    u_m = u_m[:, 1:]                        

    alpha_l = np.expand_dims(alpha_l, axis=0)
    p = np.expand_dims(p, axis=0)
    u_m = np.expand_dims(u_m, axis=0)

    state = np.concatenate((alpha_l, p, u_m), axis=0)

    return state

def pars_dict_to_array_multi_phase(pars_dict):
    pars = np.zeros((2,))

    pars[0] = pars_dict['leak_location']
    pars[1] = pars_dict['leak_size']    

    return pars


def save_data_single_phase(idx, pars, state):
    np.save(f'data/raw_data/training_data/pars/sample_{idx}.npy', pars)
    np.save(f'data/raw_data/training_data/state/sample_{idx}.npy', state)

    print('Saved data for sample {}'.format(idx))

    return None

#@ray.remote(num_returns=1)
def simulate_pipeflow(
    PDE_model: BaseModel,
    PDE_args: dict,
    model_parameters: dict = None,
    parameters_of_interest: dict = None,
    phase: str = 'single',
    idx: int = 0
    ):

    steady_state = PDE_args['steady_state']
    PDE_args.pop('steady_state')

    PDE_model = PDE_model(
        **PDE_args,
        model_parameters=model_parameters
        )
    
    if parameters_of_interest is not None:
        PDE_model.update_parameters(parameters_of_interest)
    

    init = PDE_model.initial_condition(PDE_model.DG_vars.x.flatten('F'))

    t_final = 5000.
    sol, t_vec = PDE_model.solve(
        t=0, 
        q_init=init, 
        t_final=t_final, 
        steady_state_args=steady_state,
        print_progress=True
        )

    if phase == 'single':
        state = sol_to_state_single_phase(sol, t_vec, PDE_model)
        pars = pars_dict_to_array_single_phase(parameters_of_interest)

        save_data_single_phase(
            idx=idx,
            pars=pars,
            state=state
        )




    elif phase == 'multi':
        state = sol_to_state_multi_phase(sol, t_vec, PDE_model)
        pars = pars_dict_to_array_multi_phase(parameters_of_interest)
        x = np.linspace(PDE_model.DG_vars.x[0, 0], PDE_model.DG_vars.x[-1, -1], 512)

        plt.figure()
        plt.plot(x, state[-1, :, 50])
        plt.plot(x, state[-1, :, 150])
        plt.plot(x, state[-1, :, 200])
        plt.plot(x, state[-1, :, -1])
        plt.show()


        plt.figure()
        plt.plot(x, state[0, :, 50])
        plt.plot(x, state[0, :, 150])
        plt.plot(x, state[0, :, 200])
        plt.plot(x, state[0, :, -1])
        plt.show()
        pdb.set_trace()

    return None

