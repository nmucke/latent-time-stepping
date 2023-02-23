import time
from attr import dataclass
import ray
from data_assimilation.pytorch.PDE_models import PipeflowEquations
import numpy as np
import matplotlib.pyplot as plt
import pdb
from matplotlib.animation import FuncAnimation
from data_assimilation.pytorch.particle_filter import ParticleFilter
from data_assimilation.pytorch.test_cases.pipe_flow_equation import (
    TrueSolution,
    ObservationOperator,
    PipeFlowForwardModel,
)
import torch
from scipy.stats import gaussian_kde

torch.set_default_dtype(torch.float32)

# set seed
torch.manual_seed(0)
np.random.seed(0)

CASE = 0
TRUE_SOL_PATH = f'data/raw_data/test_data/state/sample_{CASE}.npy'
TRUE_PARS_PATH = f'data/raw_data/test_data/pars/sample_{CASE}.npy'

PREPROCESSOR_PATH = 'data/processed_data/trained_preprocessor.pt'
preprocessor = torch.load(PREPROCESSOR_PATH)

LATENT_DATA_PATH = 'data/processed_data/training_data/latent_states.pt'
PARS_DATA_PATH = 'data/processed_data/training_data/pars.pt'

xmin=0.
xmax=1000

'''
BC_params = {
    'type':'dirichlet',
    'treatment': 'naive',
    'numerical_flux': 'lax_friedrichs',
}

numerical_flux_type = 'lax_friedrichs'
numerical_flux_params = {
    'alpha': 0.0,
}
stabilizer_type = 'filter'
stabilizer_params = {
    'num_modes_to_filter': 10,
    'filter_order': 6,
}

stabilizer_type = 'slope_limiter'
stabilizer_params = {
    'second_derivative_upper_bound': 1e-8,
}

step_size = 0.05
time_integrator_type = 'implicit_euler'
time_integrator_params = {
    'step_size': step_size,
    'newton_params':{
        'solver': 'krylov',
        'max_newton_iter': 200,
        'newton_tol': 1e-5
        }
    }

steady_state = {
    'newton_params':{
        'solver': 'direct',
        'max_newton_iter': 200,
        'newton_tol': 1e-5
    }
}

polynomial_type='legendre'
num_states=2

polynomial_order=2
num_elements=50

pipe_DG = PipeflowEquations(
    xmin=xmin,
    xmax=xmax,
    num_elements=num_elements,
    polynomial_order=polynomial_order,
    polynomial_type=polynomial_type,
    num_states=num_states,
    #steady_state=steady_state,
    BC_params=BC_params,
    stabilizer_type=stabilizer_type, 
    stabilizer_params=stabilizer_params,
    time_integrator_type=time_integrator_type,
    time_integrator_params=time_integrator_params, 
    numerical_flux_type=numerical_flux_type,
    numerical_flux_params=numerical_flux_params,
    )
'''

def get_true_sol(
    true_state, 
    true_pars, 
    observation_operator, 
    t_vec, 
    obs_times_idx, 
    observation_params
    ):
    """Get the true solution."""

    '''
    #pipe_DG.model_params['leak_location'] = true_pars[0]
    #pipe_DG.model_params['leak_size'] = true_pars[1]

    pipe_DG.update_leak(Cd=true_pars[1], leak_location=true_pars[0])

    true_sol, t_vec = pipe_DG.solve(
        t=t_vec[0],
        t_final=t_vec[-1],
        q_init=true_state_init,
        print_progress=False,
        )

    x_grid = np.linspace(xmin, xmax, 256)

    true_sol = true_sol[: , :, 0::4]
    t_vec = t_vec[0::4]


    true_sol_interp = np.zeros((true_sol.shape[0], x_grid.shape[0], true_sol.shape[-1]))

    for i in range(true_sol.shape[0]):
        for t in range(true_sol.shape[-1]):
            true_sol_interp[i, :, t] = pipe_DG.evaluate_solution(
                x=x_grid,
                sol_nodal=true_sol[i, :, t],
                )
    
    rho = true_sol_interp[0] / pipe_DG.A
    u = true_sol_interp[1] / rho / pipe_DG.A

    true_sol_interp[0] = rho
    true_sol_interp[1] = u

    true_sol_interp = torch.tensor(true_sol_interp, dtype=torch.float32)

    #pars = preprocessor.transform_pars(pars)
    true_sol_interp = preprocessor.transform_state(true_sol_interp)

    true_pars = torch.tensor(true_pars, dtype=torch.float32)
    true_pars = preprocessor.transform_pars(true_pars)
    '''

    x_grid = torch.linspace(xmin, xmax, 256)

    true_sol = TrueSolution(
        x_vec=x_grid,
        t_vec=np.array(t_vec),
        sol=true_state,
        pars=true_pars,
        obs_times_idx=obs_times_idx,
        observation_operator=observation_operator,
        observation_noise_params=observation_params,
        )

    return true_sol

observation_operator_params = {
    'observation_index': np.arange(0, 256, 32)
}
model_error_params = {
    'state_std': torch.Tensor([0.001]),
    'pars_std': torch.Tensor([0.2, 0.2]),
    'smoothing_factor': torch.Tensor([0.0001, 0.1]),
    #0.0001
}
particle_filter_params = {
    'num_particles': 5000,
}
observation_params = {
    'std': .02,
}
likelihood_params = {
    'std': .02,
}

save_string = 'particle_filter_solution_latent'

AE_TYPE = "AE"
AE_string = f"trained_models/autoencoders/{AE_TYPE}"
time_stepper_string = f"trained_models/time_steppers/time_stepping"

if AE_TYPE == "AE":
    time_stepper_string += "_AE"
    save_string += "_AE"
AE_string += ".pt"
AE = torch.load(AE_string)
AE = AE.to('cpu')
AE.eval()

time_stepper_string += ".pt"
time_stepper = torch.load(time_stepper_string)
time_stepper = time_stepper.to('cpu')
time_stepper.eval()

def main(AE=AE):

    observation_operator = ObservationOperator(
        params=observation_operator_params
        )

    t_range = [0, 100.]

    true_pars = torch.tensor(np.load(TRUE_PARS_PATH))
    true_pars = preprocessor.transform_pars(true_pars)

    true_state = torch.tensor(np.load(TRUE_SOL_PATH))
    true_state = true_state[:, :, 0::4]

    true_state = preprocessor.transform_state(true_state)

    t_vec = torch.linspace(t_range[0], t_range[-1], 501)
    t_vec_ids = torch.arange(t_range[0], len(t_vec), 501)

    obs_times_idx = np.arange(20, len(t_vec), 10)
    obs_times_idx = np.append(np.array([16]), obs_times_idx)


    true_sol = get_true_sol(
        true_state=true_state,
        true_pars=true_pars,
        observation_operator=observation_operator,
        t_vec=t_vec,
        obs_times_idx=obs_times_idx,
        observation_params=observation_params,
        )    

    forward_model = PipeFlowForwardModel(
        time_stepping_model=time_stepper,
        AE_model=AE,
        model_error_params=model_error_params,
        )

    particle_filter = ParticleFilter(
        params=particle_filter_params,
        forward_model=forward_model,
        observation_operator=observation_operator,
        likelihood_params=likelihood_params,
        )

    init_states = torch.load(LATENT_DATA_PATH)
    init_states = init_states[:, :, 0::4]
    init_pars = torch.load(PARS_DATA_PATH)

    state_init = init_states[0:particle_filter_params['num_particles'], :, 0:16]
    pars_init = init_pars[0:particle_filter_params['num_particles']]

    t1 = time.time()
    state, pars = particle_filter.compute_filtered_solution(
        true_sol=true_sol,
        state_init=state_init,
        pars_init=pars_init,
        )
    
    '''
    HF_state = state[:, :, -1]
    pars = pars[:, :, -1]
    HF_state = AE.decoder(HF_state, pars)
    HF_state = HF_state.detach().numpy()

    '''
    HF_state = np.zeros((state.shape[0], 2, 256, state.shape[-1]))
    #state = state.to('cuda')
    #pars = pars.to('cuda')
    #AE = AE.to('cuda')
    with torch.no_grad():
        for i in range(state.shape[-1]):
            HF_state[:, :, :, i] = AE.decoder(state[:, :, i], pars[:, :, i]).detach().numpy()
    pars = pars.detach().numpy()
    t2 = time.time()
    print(f"Time: {t2 - t1}")


    np.save(f"{save_string}_state", HF_state)
    np.save(f"{save_string}_pars", pars)

    mean_state = np.mean(HF_state, axis=0)
    mean_pars = np.mean(pars, axis=0)

    std_state = np.std(HF_state, axis=0)
    std_pars = np.std(pars, axis=0)
    
    plt.figure(figsize=(20, 15))
    plt.subplot(4, 4, 1)
    plt.plot(
        np.linspace(xmin, xmax, 256), 
        mean_state[1, :, -1], 
        label='Mean sol', linewidth=3
        )
    plt.fill_between(
        np.linspace(xmin, xmax, 256),
        mean_state[1, :, -1] - std_state[1, :, -1],
        mean_state[1, :, -1] + std_state[1, :, -1],
        alpha=0.25,
        )
    plt.plot(
        np.linspace(xmin, xmax, 256), 
        true_sol.sol[1, :, true_sol.obs_times_idx[-1]], 
        label='True sol', linewidth=2
        )
    plt.plot(
        true_sol.obs_x,
        true_sol.observations[-1, :, -1],
        '.', label='observations', markersize=20
        )
        
    plt.grid()
    plt.legend()

    plt.subplot(4, 4, 2)
    plt.hist(pars[:, 0, -1], bins=30, density=True, label='advection_velocity')
    plt.axvline(true_pars[0], color='k', label='True value', linewidth=3)

    plt.subplot(4, 4, 3)
    plt.hist(pars[:, 1, -1], bins=30, density=True, label='advection_velocity')
    plt.axvline(true_pars[1], color='k', label='True value', linewidth=3)

    plt.subplot(4, 4, 4)
    plt.plot(
        range(len(std_pars[0])), 
        mean_pars[0], 
        label='Leak location', color='tab:blue', linewidth=3, linestyle='--'
        )
    plt.fill_between(
        range(len(std_pars[0])),
        mean_pars[0] - std_pars[0],
        mean_pars[0] + std_pars[0],
        alpha=0.25,
        )
    plt.plot(
        range(len(std_pars[0])), 
        true_pars[0] * np.ones(len(std_pars[0])), 
        label='True value', color='tab:blue', linewidth=3
        )

    plt.plot(
        range(len(std_pars[0])), 
        mean_pars[1], 
        label='Leak size', color='tab:orange', linewidth=3, linestyle='--'
        )
    plt.fill_between(
        range(len(std_pars[0])),
        mean_pars[1] - std_pars[1],
        mean_pars[1] + std_pars[1],
        alpha=0.25,
        )
    plt.plot(
        range(len(std_pars[0])), 
        true_pars[1] * np.ones(len(std_pars[0])), 
        label='True value', color='tab:orange', linewidth=3
        )
    plt.ylim([0, 1])

    plt.show()

    '''

    state_1 = HF_state[:, 1, 40, :]
    state_2 = HF_state[:, 1, 200, :]

    num_x_kde = 500
    state_1_kde = np.zeros((num_x_kde, state_1.shape[-1]))
    state_2_kde = np.zeros((num_x_kde, state_2.shape[-1]))
    pars_1_kde = np.zeros((num_x_kde, pars.shape[-1]))
    pars_2_kde = np.zeros((num_x_kde, pars.shape[-1]))

    for i in range(state_1.shape[-1]):
        kde_state_1 = gaussian_kde(state_1[:, i])
        kde_state_2 = gaussian_kde(state_2[:, i])

        kde_pars_1 = gaussian_kde(pars[:, 0, i])
        kde_pars_2 = gaussian_kde(pars[:, 1, i])
        
        #x_state_1 = np.linspace(np.min(state_1), np.max(state_1), num_x_kde)
        x_state_1 = np.linspace(0.75, 0.85, num_x_kde)
        x_state_2 = np.linspace(0.4, 0.7, num_x_kde)
        x_pars = np.linspace(0, 1, num_x_kde)

        state_1_kde[:, i] = kde_state_1(x_state_1)
        state_2_kde[:, i] = kde_state_2(x_state_2)

        pars_1_kde[:, i] = kde_pars_1(x_pars)
        pars_2_kde[:, i] = kde_pars_2(x_pars)

    plt.subplot(4, 4, 5)
    plt.imshow(
        state_1_kde, 
        cmap='jet', 
        aspect='auto', 
        origin='lower', 
        extent=[t_range[0], t_range[1], 0.75, 0.85]
        )
    plt.plot(t_vec[0:496], mean_state[1, 40, :], color='k', linewidth=2)
    plt.plot(t_vec[0:496], true_sol.sol[1, 40, 0:496], color='r', linewidth=1.5)
    plt.ylabel(f'Velocity at x={40/256*1000:0.1f} m')
    plt.xlabel('Time')
    plt.ylim([0.75, 0.85])
    plt.colorbar()

    plt.subplot(4, 4, 6)
    plt.imshow(
        state_2_kde,
        cmap='jet', 
        aspect='auto', 
        origin='lower', 
        extent=[t_range[0], t_range[1], 0.4, 0.75],
        )
    plt.plot(t_vec[0:496], mean_state[1, 200, :], color='k', linewidth=2)
    plt.plot(t_vec[0:496], true_sol.sol[1, 200, 0:496], color='r', linewidth=1.5)
    plt.ylabel(f'Velocity at x={200/256*1000:0.1f} m')
    plt.xlabel('Time')
    plt.ylim([0.4, 0.75])
    plt.colorbar()

    plt.subplot(4, 4, 7)
    plt.imshow(
        pars_1_kde,
        cmap='jet', 
        aspect='auto', 
        origin='lower', 
        extent=[t_range[0], t_range[1], 0, 1],
        )
    plt.plot(t_vec[0:496], true_pars[0]*np.ones(496), color='k', linewidth=3)
    plt.colorbar()

    plt.subplot(4, 4, 8)
    plt.imshow(
        pars_2_kde,
        cmap='jet', 
        aspect='auto', 
        origin='lower', 
        extent=[t_range[0], t_range[1], 0, 1],
        )
    plt.colorbar()
    plt.plot(t_vec[0:496], true_pars[1]*np.ones(496), color='k', linewidth=3)
    plt.show()
    '''

if __name__ == '__main__':

    #ray.shutdown()
    #ray.init(num_cpus=25)
    main()
    #ray.shutdown()