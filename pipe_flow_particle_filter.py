from attr import dataclass
import ray
from data_assimilation.numpy.PDE_models import PipeflowEquations
import numpy as np
import matplotlib.pyplot as plt
import pdb
from matplotlib.animation import FuncAnimation
from data_assimilation.numpy.particle_filter import ParticleFilter
from data_assimilation.numpy.test_cases.pipe_flow_equation import (
    TrueSolution,
    ObservationOperator,
    PipeFlowForwardModel,
)
from scipy.stats import gaussian_kde

xmin=0.
xmax=1000

BC_params = {
    'type':'dirichlet',
    'treatment': 'naive',
    'numerical_flux': 'lax_friedrichs',
}

numerical_flux_type = 'lax_friedrichs'
numerical_flux_params = {
    'alpha': 0.5,
}
stabilizer_type = 'slope_limiter'
stabilizer_params = {
    'second_derivative_upper_bound': 1e-4,
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
num_elements=100

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


def get_true_sol(
    true_state, 
    true_pars,
    observation_operator, 
    t_vec, 
    obs_times_idx, 
    observation_params
    ):
    """Get the true solution."""

    #pipe_DG.model_params['leak_location'] = true_pars[0]
    #pipe_DG.model_params['leak_size'] = true_pars[1]

    '''
    pipe_DG.update_leak(Cd=true_pars[1], leak_location=true_pars[0])

    true_sol, t_vec = pipe_DG.solve(
        t=t_vec[0],
        t_final=t_vec[-1],
        q_init=true_state_init,
        print_progress=False,
        )
    

    u = np.zeros((len(x), len(t_vec)))
    rho = np.zeros((len(x), len(t_vec)))
    for t in range(true_sol.shape[-1]):
        rho[:, t] = pipe_DG.evaluate_solution(x, sol_nodal=true_sol[0, :, t])
        u[:, t] = pipe_DG.evaluate_solution(x, sol_nodal=true_sol[1, :, t])
    rho = rho / pipe_DG.A
    u = u / rho / pipe_DG.A

    true_sol = np.stack([rho, u], axis=0)
    '''
    x = np.linspace(0, 1000, 256)

    true_sol = TrueSolution(
        x_vec=x,
        t_vec=np.array(t_vec),
        sol=true_state,
        pars=true_pars,
        obs_times_idx=obs_times_idx,
        observation_operator=observation_operator,
        observation_noise_params=observation_params,
        )

    return true_sol

observation_operator_params = {
    #'observation_index': np.arange(0, pipe_DG.DG_vars.Np * pipe_DG.DG_vars.K, 50)
    'observation_index': np.arange(0, 256, 32)
}
model_error_params = {
    'state_std': [.001, .001],
    'pars_std': [30, 1e-2],
    'smoothing_factor': 0.8,
}
particle_filter_params = {
    'num_particles': 10000,
}
observation_params = {
    'std': .04,
}
likelihood_params = {
    'std': .04,
}

CASE = 0
TRUE_SOL_PATH = f'data/raw_data/test_data/state/sample_{CASE}.npy'
TRUE_PARS_PATH = f'data/raw_data/test_data/pars/sample_{CASE}.npy'

SAVE_PATH = f'particle_filter_results/high_fidelity'

true_pars = np.load(TRUE_PARS_PATH)
true_state = np.load(TRUE_SOL_PATH)
def main():

    x = np.linspace(0, 1000, 256)

    observation_operator = ObservationOperator(
        params=observation_operator_params
        )

    #true_state_init = pipe_DG.initial_condition(pipe_DG.DG_vars.x.flatten('F'))
    #true_pars = np.array([500, 1.5])


    t_range = [0, 100.]
    t_vec = np.arange(t_range[0], t_range[1], step_size)

    obs_times_idx = np.arange(20*4, len(t_vec), 40)
    obs_times_idx = np.append(np.array([16*4]), obs_times_idx)

    true_sol = get_true_sol(
        true_state=true_state,
        true_pars=true_pars,
        observation_operator=observation_operator,
        t_vec=t_vec,
        obs_times_idx=obs_times_idx,
        observation_params=observation_params,
        )    

    forward_model = PipeFlowForwardModel(
        model=pipe_DG,
        model_error_params=model_error_params,
        )

    particle_filter = ParticleFilter(
        params=particle_filter_params,
        forward_model=forward_model,
        observation_operator=observation_operator,
        likelihood_params=likelihood_params,
        )
    
    state_init = pipe_DG.initial_condition(pipe_DG.DG_vars.x.flatten('F'))
    pars_init = np.array([600, 1.5])
    particle_filter.compute_filtered_solution(
        true_sol=true_sol,
        state_init=state_init,
        pars_init=pars_init,
        save_path=SAVE_PATH,
        )

    '''
    state, pars = particle_filter.compute_filtered_solution(
        true_sol=true_sol,
        state_init=state_init,
        pars_init=pars_init,
        save_path=None # SAVE_PATH,
        )

    #np.save(f'HF_particle_filter_solution_state.npy', state)
    #np.save(f'HF_particle_filter_solution_pars.npy', pars)

    mean_state = np.mean(state, axis=0)
    mean_pars = np.mean(pars, axis=0)

    std_state = np.std(state, axis=0)
    std_pars = np.std(pars, axis=0)
    
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.plot(x, mean_state[1, :, -1], label='Mean sol', linewidth=3)
    plt.fill_between(
        x,
        mean_state[1, :, -1] - std_state[1, :, -1],
        mean_state[1, :, -1] + std_state[1, :, -1],
        alpha=0.25,
        )
    plt.plot(
        x, 
        true_sol.sol[1, :, true_sol.obs_times_idx[-1]], 
        label='True sol', linewidth=2
        )
    plt.plot(
        true_sol.obs_x,
        true_sol.observations[-1, :, -1],
        '.',
        label='observations', markersize=20)
    plt.grid()
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.hist(pars[:, 0, -1], bins=30, density='Particle filter', label='advection_velocity')
    plt.axvline(true_pars[0], color='k', label='True value', linewidth=3)

    plt.subplot(1, 4, 3)
    plt.hist(pars[:, 1, -1], bins=30, density='Particle filter', label='advection_velocity')
    plt.axvline(true_pars[1], color='k', label='True value', linewidth=3)

    plt.subplot(1, 4, 4)
    plt.plot(
        range(len(std_pars[0])), 
        mean_pars[0], 
        label='advection_velocity', 
        color='tab:blue', 
        linewidth=3, 
        linestyle='--')
    plt.fill_between(
        range(len(std_pars[0])),
        mean_pars[0] - std_pars[0],
        mean_pars[0] + std_pars[0],
        alpha=0.25,
        )
    plt.plot(range(len(std_pars[0])), true_pars[0] * np.ones(len(std_pars[0])), label='True value', color='tab:blue', linewidth=3)
    plt.savefig('particle_filter_solution.png')

    plt.show()
    '''

    '''

    plt.figure(figsize=(20, 5))

    num_steps = state.shape[-1]
    t_vec = np.linspace(0, t_range, num_steps) 

    state_1 = state[:, 1, 40, :]
    state_2 = state[:, 1, 100, :]

    num_x_kde = 500
    state_1_kde = np.zeros((num_x_kde, state_1.shape[-1]))
    state_2_kde = np.zeros((num_x_kde, state_2.shape[-1]))
    pars_1_kde = np.zeros((num_x_kde, len(obs_times_idx)))
    pars_2_kde = np.zeros((num_x_kde, len(obs_times_idx)))

    for i in range(state_1.shape[-1]):
        kde_state_1 = gaussian_kde(state_1[:, i])
        kde_state_2 = gaussian_kde(state_2[:, i])

        x_state_1 = np.linspace(np.min(state_1), np.max(state_1), num_x_kde)
        x_state_2 = np.linspace(np.min(state_2), np.max(state_2), num_x_kde)

        state_1_kde[:, i] = kde_state_1(x_state_1)
        state_2_kde[:, i] = kde_state_2(x_state_2)

    for i in range(len(obs_times_idx)):
        kde_pars_1 = gaussian_kde(pars[:, 0, i])
        kde_pars_2 = gaussian_kde(pars[:, 1, i])

        x_pars_1 = np.linspace(0, 1000, num_x_kde)
        x_pars_2 = np.linspace(1e-4, 9e-4, num_x_kde)

        pars_1_kde[:, i] = kde_pars_1(x_pars_1)
        pars_2_kde[:, i] = kde_pars_2(x_pars_2)

    plt.subplot(1, 4, 1)
    plt.imshow(
        state_1_kde, 
        cmap='jet', 
        aspect='auto', 
        origin='lower', 
        extent=[t_range[0], t_range[-1], np.min(state_1), np.max(state_1)]
        )
    plt.plot(t_vec[0:num_steps], mean_state[1, 40, :], color='k', linewidth=2)
    plt.plot(t_vec[0:num_steps], true_sol.sol[1, 40, 0:num_steps], color='r', linewidth=1.5)
    plt.ylabel(f'Velocity at x={40/256*1000:0.1f} m')
    plt.xlabel('Time')
    plt.ylim([np.min(state_1), np.max(state_1)])
    plt.colorbar()

    plt.subplot(1, 4, 2)
    plt.imshow(
        state_2_kde,
        cmap='jet', 
        aspect='auto', 
        origin='lower', 
        extent=[t_range[0], t_range[-1], np.min(state_2), np.max(state_2)],
        )
    plt.plot(t_vec[0:num_steps], mean_state[1, 100, :], color='k', linewidth=2)
    plt.plot(t_vec[0:num_steps], true_sol.sol[1, 100, 0:num_steps], color='r', linewidth=1.5)
    plt.ylabel(f'Velocity at x={100/256*1000:0.1f} m')
    plt.xlabel('Time')
    plt.ylim([np.min(state_2), np.max(state_2)])
    plt.colorbar()

    plt.subplot(1, 4, 3)
    plt.imshow(
        pars_1_kde,
        cmap='jet', 
        aspect='auto', 
        origin='lower', 
        extent=[t_range[0], t_range[-1], 0, 1000],
        )
    plt.plot(t_vec[0:num_steps], true_pars[0]*np.ones(num_steps), color='k', linewidth=3)
    plt.colorbar()

    plt.subplot(1, 4, 4)
    plt.imshow(
        pars_2_kde,
        cmap='jet', 
        aspect='auto', 
        origin='lower', 
        extent=[t_range[0], t_range[-1], 1e-4, 9e-4],
        )
    plt.colorbar()
    plt.plot(t_vec[0:num_steps], true_pars[1]*np.ones(num_steps), color='k', linewidth=3)
    plt.savefig('particle_filter_solution_other.png')
    plt.show()
    '''

if __name__ == '__main__':

    ray.shutdown()
    ray.init(num_cpus=20)
    main()
    ray.shutdown()