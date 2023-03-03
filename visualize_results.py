from attr import dataclass
import ray
import torch
from data_assimilation.numpy.PDE_models import PipeflowEquations
import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy.stats import (
    gaussian_kde,
    wasserstein_distance
)
from matplotlib.animation import FuncAnimation
from data_assimilation.numpy.test_cases.pipe_flow_equation import ObservationOperator

from data_assimilation.pytorch.observation_operator import BaseObservationOperator
   

def animate_solution(
    true_sol, 
    HF_sol=None, 
    latent_sol=None, 
    latent_AE_sol=None,
    HF_std=None,
    latent_std=None,
    latent_AE_std=None,
    obs_index=None,
    t=range(0,10), 
    x=range(0,10), 
    save_path=None
    ):

    # initializing a figure in 
    # which the graph will be plotted
    fig = plt.figure() 
    
    # marking the x-axis and y-axis
    axis = plt.axes(xlim=(x.min(), x.max()), ylim=(3, 5)) 
    
    # initializing a line variable
    if true_sol is not None:
        line_true, = axis.plot([], [], lw = 3, color='tab:blue', label='True Solution')

    if HF_sol is not None:
        line_hf, = axis.plot([], [], lw = 3, color='tab:orange', label='High-fidelity')
    if latent_sol is not None:
        line_latent, = axis.plot([], [], lw = 3, color='tab:green', label='Transformer + WAE')
    if latent_AE_sol is not None:
        line_latent_AE, = axis.plot([], [], lw = 3, color='tab:red', label='Transformer + AE')
    
    # data which the line will 
    # contain (x, y)
    def init(): 
        line_true.set_data([], [])
        return line_true,
    
    def animate(i):
        axis.collections.clear()

        if true_sol is not None:
            y = true_sol[:, i]
            line_true.set_data(x, y)
    
        # plots a sine graph
        if HF_sol is not None:
            y = HF_sol[:, i]
            line_hf.set_data(x, y)

        if HF_std is not None:
            axis.fill_between(
                x,
                HF_sol[:, i] - HF_std[:, i],
                HF_sol[:, i] + HF_std[:, i],
                alpha=0.25,
                color='tab:orange'
            )

        if latent_sol is not None:
            y2 = latent_sol[:, i]
            line_latent.set_data(x, y2)

        if latent_std is not None:
            axis.fill_between(
                x,
                latent_sol[:, i] - latent_std[:, i],
                latent_sol[:, i] + latent_std[:, i],
                alpha=0.25,
                color='tab:green'
            )

        if latent_AE_sol is not None:
            y3 = latent_AE_sol[:, i]
            line_latent_AE.set_data(x, y3)

        if latent_AE_std is not None:
            axis.fill_between(
                x,
                latent_AE_sol[:, i] - latent_AE_std[:, i],
                latent_AE_sol[:, i] + latent_AE_std[:, i],
                alpha=0.25,
                color='tab:red'
            )
        
        if obs_index is not None:
            for idx in obs_index:
                axis.axvline(
                    x[idx],
                    color='k',
                    linestyle='-',
                    linewidth=0.5,
                )

        axis.legend(loc='upper right')
        axis.set_xlabel('Space')
        axis.set_ylabel('Velocity')
        axis.set_title(f'Time: {t[i]:.2f} seconds')
        #axis.grid()
        return line_true,
    
    anim = FuncAnimation(
        fig, 
        animate, 
        init_func=init, 
        frames=len(t), 
        interval=1, 
        blit=True
        )
    
    anim.save(f'{save_path}.gif', fps=30)

#HF_STATE_PATH = "HF_particle_filter_solution_state.npy"
#HF_STATE_PATH = "HF_state.npy"
#HF_PARS_PATH = "HF_particle_filter_solution_pars.npy"
HF_STATE_PATH = 'particle_filter_results/high_fidelity/state_ensemble_filtered_'
HF_PARS_PATH = 'particle_filter_results/high_fidelity/pars_ensemble_filtered_'

LATENT_PATH = 'particle_filter_solution_latent'
LATENT_STATE_PATH = f"{LATENT_PATH}_state.npy"
LATENT_PARS_PATH = f"{LATENT_PATH}_pars.npy"


LATENT_AE_PATH = 'particle_filter_solution_latent_AE'
LATENT_STATE_AE_PATH = f"{LATENT_AE_PATH}_state.npy"
LATENT_PARS_AE_PATH = f"{LATENT_AE_PATH}_pars.npy"

CASE = 0
TRUE_STATE_PATH = f'data/raw_data/test_data/state/sample_{CASE}.npy'
TRUE_PARS_PATH = f'data/raw_data/test_data/pars/sample_{CASE}.npy'


PREPROCESSOR_PATH = 'data/processed_data/trained_preprocessor.pt'
preprocessor = torch.load(PREPROCESSOR_PATH)

xmin=0.
xmax=1000

def main():

    obs_index = np.arange(0, 256, 32)

    # Load latent particle filter results
    latent_state = np.load(LATENT_STATE_PATH)
    latent_pars = np.load(LATENT_PARS_PATH)

    latent_state = torch.tensor(latent_state)
    latent_pars = torch.tensor(latent_pars)

    for i in range(latent_state.shape[0]):
        latent_state[i] = preprocessor.inverse_transform_state(latent_state[i])
    for i in range(latent_pars.shape[0]):
        latent_pars[i] = preprocessor.inverse_transform_pars(latent_pars[i])

    latent_state = latent_state.numpy()
    latent_pars = latent_pars.numpy()


    # Load latent particle filter results
    latent_state_AE = np.load(LATENT_STATE_AE_PATH)
    latent_pars_AE = np.load(LATENT_PARS_AE_PATH)

    latent_state_AE = torch.tensor(latent_state_AE)
    latent_pars_AE = torch.tensor(latent_pars_AE)

    for i in range(latent_state_AE.shape[0]):
        latent_state_AE[i] = preprocessor.inverse_transform_state(latent_state_AE[i])
    for i in range(latent_pars_AE.shape[0]):
        latent_pars_AE[i] = preprocessor.inverse_transform_pars(latent_pars_AE[i])

    latent_state_AE = latent_state_AE.numpy()
    latent_pars_AE = latent_pars_AE.numpy()

    true_state = np.load(TRUE_STATE_PATH)
    true_pars = np.load(TRUE_PARS_PATH)

    true_state = true_state[:, :, 0::4]

    x = np.linspace(0, 1000, 256)


    observation_operator_params = {
        'observation_index': np.arange(0, 256, 32)
    }

    observation_operator = ObservationOperator(
        params=observation_operator_params
        )
    obs = observation_operator.get_observations(true_state)
    obs = obs + np.random.normal(0, 0.04, obs.shape)
    x_obs = x[observation_operator_params['observation_index']]    


    # Load high-fidelity particle filter results
    #HF_state = []
    HF_pars = []

    HF_mean_state = []
    HF_mean_pars = []
    HF_std_state = []
    HF_std_pars = []
    for i in range(39):
        HF_state = np.load(f'{HF_STATE_PATH}{i}.npy')
        HF_pars_ = np.load(f'{HF_PARS_PATH}{i}.npy')

        HF_mean_state.append(np.mean(HF_state, axis=0))
        #HF_mean_pars.append(np.mean(HF_pars, axis=0))
        HF_std_state.append(np.std(HF_state, axis=0))
        #HF_std_pars.append(np.std(HF_pars, axis=0))

        HF_pars.append(HF_pars_)

        #HF_pars = np.load(f'{HF_PARS_PATH}{i}.npy')

        #HF_state.append(np.load(f'{HF_STATE_PATH}{i}.npy'))
        #HF_pars.append(np.load(f'{HF_PARS_PATH}{i}.npy'))
    

    HF_mean_state = np.concatenate(HF_mean_state, axis=-1)
    #HF_mean_pars = np.concatenate(HF_mean_pars, axis=-1)
    HF_std_state = np.concatenate(HF_std_state, axis=-1)
    #HF_std_pars = np.concatenate(HF_std_pars, axis=-1)
    HF_pars = np.concatenate(HF_pars, axis=-1)
    HF_pars = HF_pars[:, :, 0::4]
    
    HF_mean_state = HF_mean_state[:, :, 0::4]
    HF_mean_pars = np.mean(HF_pars, axis=0)
    HF_std_state = HF_std_state[:, :, 0::4]
    HF_std_pars = np.std(HF_pars, axis=0)
    #HF_state = np.concatenate(HF_state, axis=-1)
    #HF_pars = np.concatenate(HF_pars, axis=-1)
    #HF_state = HF_state[:, :, :, 0::4]
    #HF_pars = HF_pars[:, :, 0::4]
    #HF_state = np.load(HF_STATE_PATH)
    #HF_pars = np.load(HF_PARS_PATH)

    num_steps = np.min(
        [HF_mean_state.shape[-1], latent_state.shape[-1], latent_state_AE.shape[-1], true_state.shape[-1]]
        )
    #HF_pars = np.repeat(HF_pars, 10, axis=-1)
    #HF_pars = HF_pars[:, :, 0:latent_state.shape[-1]]
    #HF_state = HF_state[:, :, :, 0:num_steps]
    true_state = true_state[:, :, 0:num_steps]
    latent_state = latent_state[:, :, 0:num_steps]
    latent_state_AE = latent_state_AE[:, :, 0:num_steps]
    HF_mean_state = HF_mean_state[:, :, 0:num_steps]
    HF_std_state = HF_std_state[:, :, 0:num_steps]

    latent_pars = latent_pars[:, :, 0:num_steps]
    latent_pars_AE = latent_pars_AE[:, :, 0:num_steps]

    t_range = [0, 100.]
    t_vec = np.linspace(t_range[0], t_range[1], 501)
    t_vec = t_vec[0:num_steps]

    #HF_mean_state = np.mean(HF_state, axis=0)
    #HF_mean_pars = np.mean(HF_pars, axis=0)
    #HF_std_state = np.std(HF_state, axis=0)
    #HF_std_pars = np.std(HF_pars, axis=0)

    latent_mean_state = np.mean(latent_state, axis=0)
    latent_mean_pars = np.mean(latent_pars, axis=0)
    latent_std_state = np.std(latent_state, axis=0)
    latent_std_pars = np.std(latent_pars, axis=0)

    latent_mean_state_AE = np.mean(latent_state_AE, axis=0)
    latent_mean_pars_AE = np.mean(latent_pars_AE, axis=0)
    latent_std_state_AE = np.std(latent_state_AE, axis=0)
    latent_std_pars_AE = np.std(latent_pars_AE, axis=0)
    
    animate_solution(
        #HF_sol=HF_mean_state[1, :, :], 
        true_sol=true_state[1, :, :],
        #latent_sol=latent_mean_state[1, :, :],
        #latent_AE_sol=latent_mean_state_AE[1, :, :],
        #HF_std=1.96*HF_std_state[1, :, :],
        #latent_std=1.96*latent_std_state[1, :, :],
        #latent_AE_std=1.96*latent_std_state_AE[1, :, :],
        obs_index=obs_index,
        t=t_vec, 
        x=x, 
        save_path='true_sol'
        )

    num_x_kde = 100
    HF_pars_1_kde = np.zeros((num_x_kde, HF_pars.shape[-1]))
    HF_pars_2_kde = np.zeros((num_x_kde, HF_pars.shape[-1]))
    latent_pars_1_kde = np.zeros((num_x_kde, HF_pars.shape[-1]))
    latent_pars_2_kde = np.zeros((num_x_kde, HF_pars.shape[-1]))
    latent_pars_AE_1_kde = np.zeros((num_x_kde, HF_pars.shape[-1]))
    latent_pars_AE_2_kde = np.zeros((num_x_kde, HF_pars.shape[-1]))
    for i in range(HF_pars.shape[-1]):

        x_pars_1 = np.linspace(600, 800, num_x_kde)
        x_pars_2 = np.linspace(1, 2, num_x_kde)


        x_pars_1_AE = np.linspace(200, 1000, num_x_kde)

        kde_pars_1 = gaussian_kde(HF_pars[:, 0, i])
        kde_pars_2 = gaussian_kde(HF_pars[:, 1, i])
        HF_pars_1_kde[:, i] = kde_pars_1(x_pars_1)
        HF_pars_2_kde[:, i] = kde_pars_2(x_pars_2)

        kde_pars_1 = gaussian_kde(latent_pars[:, 0, i])
        kde_pars_2 = gaussian_kde(latent_pars[:, 1, i])
        latent_pars_1_kde[:, i] = kde_pars_1(x_pars_1)
        latent_pars_2_kde[:, i] = kde_pars_2(x_pars_2)

        kde_pars_1 = gaussian_kde(latent_pars_AE[:, 0, i])
        kde_pars_2 = gaussian_kde(latent_pars_AE[:, 1, i])
        latent_pars_AE_1_kde[:, i] = kde_pars_1(x_pars_1_AE)
        latent_pars_AE_2_kde[:, i] = kde_pars_2(x_pars_2)



    plot_time = 380
    plt.figure()
    plt.plot(x, HF_mean_state[1, :, plot_time], label='High-fidelity', linewidth=3, color='tab:blue')
    plt.fill_between(
        x,
        HF_mean_state[1, :, plot_time] - HF_std_state[1, :, plot_time],
        HF_mean_state[1, :, plot_time] + HF_std_state[1, :, plot_time],
        alpha=0.25,
        color='tab:blue'
        )
    plt.plot(x, latent_mean_state[1, :, plot_time], label='Transformer + WAE', linewidth=3, color='tab:green')
    plt.fill_between(
        x,
        latent_mean_state[1, :, plot_time] - latent_std_state[1, :, plot_time],
        latent_mean_state[1, :, plot_time] + latent_std_state[1, :, plot_time],
        alpha=0.25,
        color='tab:green'
        )
    
    plt.plot(x, latent_mean_state_AE[1, :, plot_time], label='Transformer + AE', linewidth=3, color='tab:red')
    plt.fill_between(
        x,
        latent_mean_state_AE[1, :, plot_time] - latent_std_state_AE[1, :, plot_time],
        latent_mean_state_AE[1, :, plot_time] + latent_std_state_AE[1, :, plot_time],
        alpha=0.25,
        color='tab:red'
        )
    
    
    plt.plot(
        x, 
        true_state[1, :, plot_time], 
        label='True State', linewidth=2, color='tab:orange'
        )
        
    plt.plot(
        x_obs,
        obs[0, :, plot_time],
        '.',
        label='observations', markersize=20,
        color='k'
        )
        
    plt.grid()
    plt.legend()
    plt.xlabel('Space')
    plt.ylabel('Velocity')
    plt.savefig(f'particle_filter_solution_t{plot_time}.pdf')
    plt.show()
    pdb.set_trace()


    wasserstein_error_pars_1 = wasserstein_distance(
        HF_pars[:, 0, -1], 
        latent_pars[:, 0, -1]
        )
    wasserstein_error_pars_2 = wasserstein_distance(
        HF_pars[:, 1, -1], 
        latent_pars[:, 1, -1]
        )


    wasserstein_error_pars_1_AE = wasserstein_distance(
        HF_pars[:, 0, -1], 
        latent_pars_AE[:, 0, -1]
        )
    wasserstein_error_pars_2_AE = wasserstein_distance(
        HF_pars[:, 1, -1], 
        latent_pars_AE[:, 1, -1]
        )

    print(f"WAE: {wasserstein_error_pars_1}")
    print(f"AE: {wasserstein_error_pars_1_AE}")

    plt.figure()
    plt.hist(HF_pars[:, 0, -1], bins=50, density=True, label='High-fidelity', color='tab:blue', alpha=0.25)
    plt.plot(x_pars_1, HF_pars_1_kde[:, -1], color='tab:blue', linewidth=1, linestyle='-')
    plt.axvline(HF_pars[:, 0, -1].mean(), color='tab:blue', linewidth=3)

    plt.hist(latent_pars[:, 0, -1], bins=50, density=True, label='Transformer + WAE', color='tab:green', alpha=0.25)
    plt.plot(x_pars_1, latent_pars_1_kde[:, -1], color='tab:green', linewidth=1, linestyle='-')
    plt.axvline(latent_pars[:, 0, -1].mean(), color='tab:green', linewidth=3)
    '''
    plt.hist(latent_pars_AE[:, 0, -1], bins=50, density=True, label='Transformer + AE', color='tab:red', alpha=0.25)
    plt.plot(x_pars_1_AE, latent_pars_AE_1_kde[:, -1], color='tab:red', linewidth=1, linestyle='-')
    plt.axvline(latent_pars_AE[:, 0, -1].mean(), color='tab:red', linewidth=3)
    '''

    plt.axvline(true_pars[0], color='k', label='True value', linewidth=3)
    plt.legend()
    plt.grid()
    #plt.title('Wasserstein distance: ' + str(wasserstein_error_pars_1))

    plt.savefig('hist_leak_location_WAE.pdf')
    #plt.savefig('hist_leak_location_all.pdf')
    plt.show()

    plt.figure()
    plt.hist(HF_pars[:, 1, -1], bins=30, density=True, label='HF Particle Filter',color='tab:blue', alpha=0.25)
    plt.plot(x_pars_2, HF_pars_2_kde[:, -1], color='tab:blue', linewidth=1, linestyle='-')
    plt.axvline(HF_pars[:, 1, -1].mean(), color='tab:blue', linewidth=3)

    plt.hist(latent_pars[:, 1, -1], bins=30, density=True, label='NN Particle Filter', color='tab:green', alpha=0.25)
    plt.plot(x_pars_2, latent_pars_2_kde[:, -1], color='tab:green', linewidth=1, linestyle='-')
    plt.axvline(latent_pars[:, 1, -1].mean(), color='tab:green', linewidth=3)

    plt.axvline(true_pars[1], color='k', label='True value', linewidth=3)
    plt.legend()
    plt.title('Wasserstein distance: ' + str(wasserstein_error_pars_2))
    plt.show()

    plt.figure()
    plt.plot(
        range(HF_std_pars.shape[-1]), 
        HF_mean_pars[0], 
        label='High-fidelity', 
        linewidth=3, 
        linestyle='-',
        color='tab:blue', 
        )
    plt.fill_between(
        range(HF_std_pars.shape[-1]),
        HF_mean_pars[0] - HF_std_pars[0],
        HF_mean_pars[0] + HF_std_pars[0],
        alpha=0.25,
        color='tab:blue'
        )
    plt.plot(
        range(len(latent_std_pars[-1])), 
        latent_mean_pars[0], 
        label='Transformer + WAE', 
        linewidth=3, 
        linestyle='-',
        color='tab:green'
        )
    plt.fill_between(
        range(len(latent_std_pars[-1])),
        latent_mean_pars[0] - latent_std_pars[0],
        latent_mean_pars[0] + latent_std_pars[0],
        alpha=0.25,
        color='tab:green'
        )

    plt.plot(
        range(len(latent_std_pars_AE[-1])),
        latent_mean_pars_AE[0],
        label='Transformer + AE',
        linewidth=3,
        linestyle='-',
        color='tab:red'
    )
    plt.fill_between(
        range(len(latent_std_pars_AE[-1])),
        latent_mean_pars_AE[0] - latent_std_pars_AE[0],
        latent_mean_pars_AE[0] + latent_std_pars_AE[0],
        alpha=0.25,
        color='tab:red'
    )

    plt.plot(
        range(len(HF_std_pars[-1])), 
        true_pars[0] * np.ones(len(HF_std_pars[0])), 
        label='True value', 
        linewidth=3,
        color='k'
        )
    #plt.savefig('particle_filter_solution.png')
    plt.legend()
    plt.grid()
    plt.savefig('time_leak_location_all.pdf')
    plt.show()



if __name__ == '__main__':
    num_samples = [100, 500, 1000, 2000, 3000, 4000, 5000]
    WAE = [14.748, 14.412, 8.489, 11.267, 10.245, 5.653, 4.795]
    AE = [177, 115.579, 120.687, 113.142, 160.437, 127.111, 125.154]
    plt.figure()
    plt.semilogy(num_samples, WAE, '.-', label='Transformer + WAE', linewidth=3, markersize=15)
    plt.semilogy(num_samples, AE, '.-', label='Transformer + AE', linewidth=3, markersize=15)
    plt.xlabel('Number of samples')
    plt.ylabel('Wasserstein distance')
    plt.legend()
    plt.grid()
    plt.savefig('leak_location_convergence.pdf')
    plt.show()
    main()

