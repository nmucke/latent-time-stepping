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
   

def animate_solution(
    sol_1, 
    sol_2=None, 
    sol_3=None, 
    std_1=None,
    std_2=None,
    std_3=None,
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
    line, = axis.plot([], [], lw = 3, color='tab:blue', label='HF Particle Filter')

    if sol_2 is not None:
        line2, = axis.plot([], [], lw = 3, color='tab:orange', label='True Solution')
    if sol_3 is not None:
        line3, = axis.plot([], [], lw = 3, color='tab:green', label='NN Particle Filter')
    
    # data which the line will 
    # contain (x, y)
    def init(): 
        line.set_data([], [])
        return line,
    
    def animate(i):
        axis.collections.clear()
    
        # plots a sine graph
        y = sol_1[:, i]
        line.set_data(x, y)

        if std_1 is not None:
            axis.fill_between(
                x,
                sol_1[:, i] - std_1[:, i],
                sol_1[:, i] + std_1[:, i],
                alpha=0.25,
                color='tab:blue'
            )

        #std.set_data(
        #    )

        if sol_2 is not None:
            y2 = sol_2[:, i]
            line2.set_data(x, y2)

        if std_2 is not None:
            axis.fill_between(
                x,
                sol_2[:, i] - std_2[:, i],
                sol_2[:, i] + std_2[:, i],
                alpha=0.25,
                color='tab:orange'
            )

        if sol_3 is not None:
            y3 = sol_3[:, i]
            line3.set_data(x, y3)

        if std_3 is not None:
            axis.fill_between(
                x,
                sol_3[:, i] - std_3[:, i],
                sol_3[:, i] + std_3[:, i],
                alpha=0.25,
                color='tab:green'
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
        return line,
    
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

LATENT_STATE_PATH = "latent_state.npy"
LATENT_PARS_PATH = "latent_pars.npy"

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

    true_state = np.load(TRUE_STATE_PATH)
    true_pars = np.load(TRUE_PARS_PATH)

    true_state = true_state[:, :, 0::4]

    x = np.linspace(0, 1000, 256)


    # Load high-fidelity particle filter results
    HF_state = []
    HF_pars = []
    for i in range(49):
        HF_state.append(np.load(f'{HF_STATE_PATH}{i}.npy'))
        HF_pars.append(np.load(f'{HF_PARS_PATH}{i}.npy'))
    
    HF_state = np.concatenate(HF_state, axis=-1)
    HF_pars = np.concatenate(HF_pars, axis=-1)
    
    HF_state = HF_state[:, :, :, 0::4]
    HF_pars = HF_pars[:, :, 0::4]
    #HF_state = np.load(HF_STATE_PATH)
    #HF_pars = np.load(HF_PARS_PATH)

    #HF_pars = np.repeat(HF_pars, 10, axis=-1)
    #HF_pars = HF_pars[:, :, 0:latent_state.shape[-1]]
    num_steps = np.min([HF_state.shape[-1], latent_state.shape[-1], true_state.shape[-1]])
    HF_state = HF_state[:, :, :, 0:num_steps]
    true_state = true_state[:, :, 0:num_steps]

    t_range = [0, 100.]
    t_vec = np.linspace(t_range[0], t_range[1], num_steps)

    HF_mean_state = np.mean(HF_state, axis=0)
    HF_mean_pars = np.mean(HF_pars, axis=0)
    HF_std_state = np.std(HF_state, axis=0)
    HF_std_pars = np.std(HF_pars, axis=0)

    latent_mean_state = np.mean(latent_state, axis=0)
    latent_mean_pars = np.mean(latent_pars, axis=0)
    latent_std_state = np.std(latent_state, axis=0)
    latent_std_pars = np.std(latent_pars, axis=0)

    animate_solution(
        sol_1=HF_mean_state[1, :, :], 
        sol_2=true_state[1, :, :],
        sol_3=latent_mean_state[1, :, :],
        std_1=1.96*HF_std_state[1, :, :],
        std_3=1.96*latent_std_state[1, :, :],
        obs_index=obs_index,
        t=t_vec, 
        x=x, 
        save_path='particle_filter_solution'
        )

    num_x_kde = 100
    HF_pars_1_kde = np.zeros((num_x_kde, HF_pars.shape[-1]))
    HF_pars_2_kde = np.zeros((num_x_kde, HF_pars.shape[-1]))
    latent_pars_1_kde = np.zeros((num_x_kde, HF_pars.shape[-1]))
    latent_pars_2_kde = np.zeros((num_x_kde, HF_pars.shape[-1]))
    for i in range(HF_pars.shape[-1]):

        x_pars_1 = np.linspace(600, 800, num_x_kde)
        x_pars_2 = np.linspace(1, 2, num_x_kde)

        kde_pars_1 = gaussian_kde(HF_pars[:, 0, i])
        kde_pars_2 = gaussian_kde(HF_pars[:, 1, i])
        HF_pars_1_kde[:, i] = kde_pars_1(x_pars_1)
        HF_pars_2_kde[:, i] = kde_pars_2(x_pars_2)

        kde_pars_1 = gaussian_kde(latent_pars[:, 0, i])
        kde_pars_2 = gaussian_kde(latent_pars[:, 1, i])
        latent_pars_1_kde[:, i] = kde_pars_1(x_pars_1)
        latent_pars_2_kde[:, i] = kde_pars_2(x_pars_2)


    plot_time = 300
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.plot(x, HF_mean_state[1, :, plot_time], label='HF Particle Filter', linewidth=3, color='tab:blue')
    plt.fill_between(
        x,
        HF_mean_state[1, :, plot_time] - HF_std_state[1, :, plot_time],
        HF_mean_state[1, :, plot_time] + HF_std_state[1, :, plot_time],
        alpha=0.25,
        color='tab:blue'
        )
    plt.plot(x, latent_mean_state[1, :, plot_time], label='HF Particle Filter', linewidth=3, color='tab:green')
    plt.fill_between(
        x,
        latent_mean_state[1, :, plot_time] - latent_std_state[1, :, plot_time],
        latent_mean_state[1, :, plot_time] + latent_std_state[1, :, plot_time],
        alpha=0.25,
        color='tab:green'
        )
    
    plt.plot(
        x, 
        true_state[1, :, plot_time], 
        label='True State', linewidth=2, color='tab:orange'
        )
    '''
    plt.plot(
        true_sol.obs_x,
        true_sol.observations[-1, :, -1],
        '.',
        label='observations', markersize=20)
    plt.grid()
    plt.legend()
    '''


    wasserstein_error_pars_1 = wasserstein_distance(
        HF_pars[:, 0, -1], 
        latent_pars[:, 0, -1]
        )
    wasserstein_error_pars_2 = wasserstein_distance(
        HF_pars[:, 1, -1], 
        latent_pars[:, 1, -1]
        )


    plt.subplot(1, 4, 2)
    plt.hist(HF_pars[:, 0, -1], bins=30, density=True, label='HF Particle Filter', color='tab:blue', alpha=0.25)
    plt.plot(x_pars_1, HF_pars_1_kde[:, -1], color='tab:blue', linewidth=1, linestyle='-')
    plt.axvline(HF_pars[:, 0, -1].mean(), color='tab:blue', linewidth=3)

    plt.hist(latent_pars[:, 0, -1], bins=30, density=True, label='NN Particle Filter', color='tab:green', alpha=0.25)
    plt.plot(x_pars_1, latent_pars_1_kde[:, -1], color='tab:green', linewidth=1, linestyle='-')
    plt.axvline(latent_pars[:, 0, -1].mean(), color='tab:green', linewidth=3)

    plt.axvline(true_pars[0], color='k', label='True value', linewidth=3)
    plt.legend()
    plt.title('Wasserstein distance: ' + str(wasserstein_error_pars_1))

    plt.subplot(1, 4, 3)
    plt.hist(HF_pars[:, 1, -1], bins=30, density=True, label='HF Particle Filter',color='tab:blue', alpha=0.25)
    plt.plot(x_pars_2, HF_pars_2_kde[:, -1], color='tab:blue', linewidth=1, linestyle='-')
    plt.axvline(HF_pars[:, 1, -1].mean(), color='tab:blue', linewidth=3)

    plt.hist(latent_pars[:, 1, -1], bins=30, density=True, label='NN Particle Filter', color='tab:green', alpha=0.25)
    plt.plot(x_pars_2, latent_pars_2_kde[:, -1], color='tab:green', linewidth=1, linestyle='-')
    plt.axvline(latent_pars[:, 1, -1].mean(), color='tab:green', linewidth=3)

    plt.axvline(true_pars[1], color='k', label='True value', linewidth=3)
    plt.legend()
    plt.title('Wasserstein distance: ' + str(wasserstein_error_pars_2))

    plt.subplot(1, 4, 4)
    plt.plot(
        range(len(HF_std_pars[0])), 
        HF_mean_pars[0], 
        label='HF Particle Filter', 
        linewidth=3, 
        linestyle='-',
        color='tab:blue', 
        )
    plt.fill_between(
        range(len(HF_std_pars[0])),
        HF_mean_pars[0] - HF_std_pars[0],
        HF_mean_pars[0] + HF_std_pars[0],
        alpha=0.25,
        color='tab:blue'
        )
    plt.plot(
        range(len(latent_std_pars[0])), 
        latent_mean_pars[0], 
        label='NN Particle Filter', 
        linewidth=3, 
        linestyle='-',
        color='tab:green'
        )
    plt.fill_between(
        range(len(latent_std_pars[0])),
        latent_mean_pars[0] - latent_std_pars[0],
        latent_mean_pars[0] + latent_std_pars[0],
        alpha=0.25,
        color='tab:green'
        )
    plt.plot(
        range(len(HF_std_pars[0])), 
        true_pars[0] * np.ones(len(HF_std_pars[0])), 
        label='True value', 
        linewidth=3,
        color='k'
        )
    #plt.savefig('particle_filter_solution.png')
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == '__main__':

    main()

