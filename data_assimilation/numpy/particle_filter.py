from abc import abstractmethod
import os
from discontinuous_galerkin.base.base_model import BaseModel
import numpy as np
from attr import dataclass
import pdb
import matplotlib.pyplot as plt
from tqdm import tqdm
import ray
from scipy.stats import norm
import time

from data_assimilation.numpy.model_error import BaseModelError
from data_assimilation.numpy.forward_model import BaseForwardModel
from data_assimilation.numpy.observation_operator import BaseObservationOperator


def compute_prior_particle(
    forward_model, 
    t_range,
    state, 
    pars,
    add_error=True
    ):
    """Compute the prior."""

    if add_error:
        state, pars = \
            forward_model.model_error.add_model_error(
                state=state, 
                pars=pars
                )

    forward_model.update_params(pars=pars)

    state, t_vec = \
        forward_model.compute_forward_model(
            t_range=t_range, 
            state=state
            )
    
    return state, pars, t_vec

@ray.remote(num_returns=3)
def compute_prior_particle_ray(
    forward_model, 
    t_range,
    state, 
    pars,
    add_error
    ):
    """Compute the prior."""

    return compute_prior_particle(
        forward_model=forward_model, 
        t_range=t_range,
        state=state, 
        pars=pars,
        add_error=add_error
        )

@ray.remote
def compute_likelihood_ray(
    state: np.ndarray,
    observation: np.ndarray,
    observation_operator: BaseObservationOperator,
    likelihood_params: dict,
    ) -> np.ndarray:
    """Compute the likelihood."""
    
    return compute_likelihood(
        state=state,
        observation=observation,
        observation_operator=observation_operator,
        likelihood_params=likelihood_params,
    )

def compute_likelihood(
    state: np.ndarray,
    observation: np.ndarray,
    observation_operator: BaseObservationOperator,
    likelihood_params: dict,
    ) -> np.ndarray:
    """Compute the likelihood."""

    model_observations = observation_operator.get_observations(state=state)

    residual = observation - model_observations

    likelihood_norm = norm(loc=0, scale=likelihood_params['std'])

    likelihood = likelihood_norm.pdf(residual).sum()
    '''
    likelihood = np.exp(-0.5/likelihood_params['std']/likelihood_params['std'] * residual_norm)
    likelihood = np.exp(-0.5/10/10 * residual_norm)
    likelihood = likelihood / np.sqrt(2*np.pi*likelihood_params['std']**2)/residual.shape[1]
    '''

    return likelihood

@ray.remote
def get_regular_grid_state_ray(state, x, to_regular_grid, A):
    """Get the regular grid state."""

    return get_regular_grid_state(
        state=state,
        x=x,
        to_regular_grid=to_regular_grid,
        A=A,
    )

def get_regular_grid_state(state, x, to_regular_grid, A):
    """Get the regular grid state."""


    state_regular_grid = np.zeros(
        (state.shape[0], x.shape[0], state.shape[-1])
        )
    for t in range(state.shape[-1]):
        state_regular_grid[0, :, t] = to_regular_grid(
            x, sol_nodal=state[0, :, t]
            )
            
        state_regular_grid[1, :, t] = to_regular_grid(
            x, sol_nodal=state[1, :, t]
            )

        state_regular_grid[0, :, t] = \
            state_regular_grid[0, :, t] / A
        state_regular_grid[1, :, t] = \
            state_regular_grid[1, :, t] / state_regular_grid[0, :, t] / A

    return state_regular_grid

class ParticleFilter():

    def __init__(
        self,
        params: dict,
        forward_model: BaseForwardModel,
        observation_operator: BaseObservationOperator,
        likelihood_params: dict,
        ) -> None:

        self.params = params
        self.forward_model = forward_model
        self.observation_operator = observation_operator
        self.likelihood_params = likelihood_params 
        self.x = np.linspace(0, 1000, 256)

        self.ESS_threshold = self.params['num_particles'] / 2

    def _get_regular_grid_state(self, state):
        
        if ray.is_initialized():
            state_regular_grid = []

            for i in range(state.shape[0]):

                state_regular_grid.append(get_regular_grid_state_ray.remote(
                    state=state[i], 
                    x=self.x, 
                    to_regular_grid=self.forward_model.model.evaluate_solution, 
                    A=self.forward_model.model.A
                ))

            state_regular_grid = ray.get(state_regular_grid)
            state_regular_grid = np.stack(state_regular_grid, axis=0)

        else:
            state_regular_grid = np.zeros(
                (state.shape[0], state.shape[1], self.x.shape[0], state.shape[-1])
                )
            for i in range(state.shape[0]):
                state_regular_grid[i] = get_regular_grid_state(
                    state=state[i], 
                    x=self.x, 
                    to_regular_grid=self.forward_model.model.evaluate_solution, 
                    A=self.forward_model.model.A
                    )

        return state_regular_grid



    def _update_weights(self, likelihood, weights):
        """Update the weights of the particles."""
        
        weights = weights * likelihood
        weights = weights / weights.sum()

        ESS = 1 / np.sum(weights**2)

        return weights, ESS

    def _restart_weights(self, ):
        """Restart the weights of the particles."""
        
        return np.ones(self.params['num_particles']) / self.params['num_particles']

    def _compute_initial_ensemble(self, state_init, pars_init):
        """Compute the initial ensemble."""
        
        state_init_ensemble = np.repeat(
            np.expand_dims(state_init, 0), 
            self.params['num_particles'], 
            axis=0
            )
        
        pars_init_ensemble = np.repeat(
            np.expand_dims(pars_init, 0), 
            self.params['num_particles'], 
            axis=0
            )
        
        for i in range(self.params['num_particles']):
            state_init_ensemble[i], pars_init_ensemble[i] = \
                self.forward_model.model_error.get_initial_ensemble(
                    state=state_init_ensemble[i], 
                    pars=pars_init_ensemble[i]
                    )

        '''
        self.forward_model.model_error.add_model_error(
            state=state_init_ensemble[i], 
            pars=pars_init_ensemble[i]
            )
        '''

        return state_init_ensemble, pars_init_ensemble   

    def _compute_prior_particles(self, t_range, state, pars, add_error=True):
        """Compute the prior particles."""

        if ray.is_initialized():
            state_ensemble = []
            pars_ensemble = []
            for i in range(self.params['num_particles']):
                particle_state, particle_pars, t_vec = \
                    compute_prior_particle_ray.remote(
                        forward_model=self.forward_model, 
                        t_range=t_range,
                        state=state[i], 
                        pars=pars[i],
                        add_error=add_error
                        )
                state_ensemble.append(particle_state)
                pars_ensemble.append(particle_pars)
                
            state_ensemble = ray.get(state_ensemble)
            pars_ensemble = ray.get(pars_ensemble)

            state_ensemble = np.asarray(state_ensemble)
            pars_ensemble = np.asarray(pars_ensemble)
        else:
            state_ensemble = np.zeros((*state.shape, 1))
            pars_ensemble = np.zeros(pars.shape)
            for i in range(self.params['num_particles']):
                particle_state, particle_pars, t_vec = compute_prior_particle(
                        forward_model=self.forward_model, 
                        t_range=t_range,
                        state=state[i], 
                        pars=pars[i],
                        add_error=add_error
                        )
                state_ensemble[i] = particle_state[:, :, -1:]
                pars_ensemble[i] = particle_pars

        return state_ensemble, pars_ensemble

    def _compute_likelihood(self, state_ensemble, observation):
        """Compute the model likelihood."""

        if False: # ray.is_initialized():
            likelihood_list = []
            for i in range(self.params['num_particles']):
                likelihood = compute_likelihood_ray.remote(
                    state=state_ensemble[i], 
                    observation=observation,
                    observation_operator=self.observation_operator,
                    likelihood_params=self.likelihood_params,
                    )
                likelihood_list.append(likelihood)
            likelihood = ray.get(likelihood_list)
        else:
            likelihood = np.zeros(self.params['num_particles'])
            for i in range(self.params['num_particles']):
                likelihood[i] = compute_likelihood(
                    state=state_ensemble[i], 
                    observation=observation,
                    observation_operator=self.observation_operator,
                    likelihood_params=self.likelihood_params,
                    )
        
        return likelihood
    

    def _get_resampled_particles(self, state_ensemble, pars_ensemble, weights):
        """Compute the posterior."""
        
        resampled_ids = np.random.multinomial(
            n=self.params['num_particles'],
            pvals=weights,
        )
        indeces = np.repeat(
            np.arange(self.params['num_particles']),
            resampled_ids
        )

        return state_ensemble[indeces], pars_ensemble[indeces], indeces
    
    def compute_filtered_solution(
        self,
        true_sol,
        state_init,
        pars_init,
        save_path=None,
    ):
        """Compute the filtered solution."""


        weights = self._restart_weights()
        
        self.forward_model.model_error.initialize_model_error_distribution(
            state=state_init,
            pars=pars_init
            )
            
        state_ensemble, pars_ensemble = self._compute_initial_ensemble(
                state_init=state_init,
                pars_init=pars_init
            )

        state_ensemble = np.expand_dims(state_ensemble, -1)
        pars_ensemble = np.expand_dims(pars_ensemble, -1)
        
        p_bar = tqdm(
            enumerate(zip(true_sol.obs_t[:-1], true_sol.obs_t[1:])),
            total=len(true_sol.obs_t[1:]),
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
            )

        # Compute the prior particles
        state_ensemble, pars_ensemble = self._compute_prior_particles(
            t_range=[0, true_sol.obs_t[0]],
            state=state_ensemble[:, :, :, -1],
            pars=pars_ensemble[:, :, -1],
            add_error=False
        )
        #state_ensemble = state_ensemble[:, :, :, -1:]
        pars_ensemble = np.expand_dims(pars_ensemble, -1)
        
        state_ensemble_filtered = self._get_regular_grid_state(state_ensemble)
        pars_ensemble_filtered = pars_ensemble.copy()

        if save_path is not None:
            np.save(
                f'{save_path}/state_ensemble_filtered_0.npy',
                state_ensemble_filtered
                )
            np.save(
                f'{save_path}/pars_ensemble_filtered_0.npy',
                np.repeat(pars_ensemble, state_ensemble_filtered.shape[-1], axis=-1)
                )
        for i, (t_old, t_new) in p_bar:

            self.forward_model.model_error.update_model_error_distribution(
                state=state_ensemble[:, :, :, -1],
                pars=pars_ensemble[:, :, -1],
                weights=weights
                )
            
            # Compute the prior particles
            state_ensemble, pars_ensemble = self._compute_prior_particles(
                t_range=[t_old, t_new],
                state=state_ensemble[:, :, :, -1],
                pars=pars_ensemble[:, :, -1]
            )
            pars_ensemble = np.expand_dims(pars_ensemble, axis=-1)
            
            state_ensemble_regular_grid = self._get_regular_grid_state(
                state_ensemble
                )

            # Compute the likelihood
            likelihood = self._compute_likelihood(
                state_ensemble_regular_grid[:, :, :, -1], 
                observation=true_sol.observations[:, :, i+1]
                )
            
            weights, ESS = self._update_weights(
                likelihood=likelihood, 
                weights=weights
                )
            if ESS < self.ESS_threshold:
                state_ensemble_regular_grid, pars_ensemble, ids = \
                    self._get_resampled_particles(
                        state_ensemble=state_ensemble_regular_grid,
                        pars_ensemble=pars_ensemble,
                        weights=weights
                    )
                state_ensemble =  state_ensemble[ids]
                weights = self._restart_weights()

                resample = True
            else:
                resample = False
            
            
            if save_path is not None:
                np.save(
                    f'{save_path}/state_ensemble_filtered_{i+1}.npy',
                    state_ensemble_regular_grid
                    )
                np.save(
                    f'{save_path}/pars_ensemble_filtered_{i+1}.npy',
                    np.repeat(pars_ensemble, state_ensemble_regular_grid.shape[-1], axis=-1)
                    )
            else:
                state_ensemble_filtered = np.concatenate(
                    (state_ensemble_filtered, state_ensemble_regular_grid), 
                    axis=-1
                    )
                pars_ensemble_filtered = np.concatenate(
                    (pars_ensemble_filtered, pars_ensemble), 
                    axis=-1
                    )
            
            p_bar.set_postfix({'Resample': resample})
            
            '''
            plt.figure()
            for j in range(100):
                plt.plot(state_ensemble_filtered[j, 1, :, -1])
                plt.plot(true_sol.sol[1, :, true_sol.obs_times_idx[i+1]], linewidth=4, color='k')
            plt.show()

            pdb.set_trace()
            '''
        if save_path is not None:
            return None
        else:
            return state_ensemble_filtered, pars_ensemble_filtered
    