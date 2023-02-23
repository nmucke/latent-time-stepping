from abc import abstractmethod
from discontinuous_galerkin.base.base_model import BaseModel
import numpy as np
from attr import dataclass
import pdb
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import ray
from scipy.stats import norm

from data_assimilation.numpy.model_error import BaseModelError
from data_assimilation.numpy.forward_model import BaseForwardModel
from data_assimilation.numpy.observation_operator import BaseObservationOperator


def compute_prior_particle(
    forward_model, 
    t_range,
    state, 
    pars,
    ):
    """Compute the prior."""

    for i in range(state.shape[0]):
        state[i], pars[i] = \
            forward_model.model_error.add_model_error(
                state=state[i], 
                pars=pars[i],
                )

    #forward_model.time_stepping_model.to('cuda')
    #state = state.to('cuda')
    #pars = pars.to('cuda')
    state, t_vec = \
        forward_model.compute_forward_model(
            t_range=t_range, 
            state=state,
            pars=pars
            )
    #state = state.to('cpu').detach()
    #pars = pars.to('cpu').detach()
    state = state.detach()
    pars = pars.detach()
    return state, pars, t_vec

@ray.remote(num_returns=3)
def compute_prior_particle_ray(
    forward_model, 
    t_range,
    state, 
    pars
    ):
    """Compute the prior."""

    return compute_prior_particle(
        forward_model=forward_model, 
        t_range=t_range,
        state=state, 
        pars=pars
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

    log_likelihood_dist = torch.distributions.Normal(
        loc=torch.zeros_like(residual),
        scale=likelihood_params['std']
    )

    likelihood = log_likelihood_dist.log_prob(residual)
    likelihood = torch.exp(likelihood).sum()

    return likelihood

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

        self.ESS_threshold = self.params['num_particles'] / 2

    def _update_weights(self, likelihood, weights):
        """Update the weights of the particles."""
        
        weights = weights * likelihood
        weights = weights / weights.sum()

        ESS = 1 / torch.sum(weights**2)

        weights = weights.type(torch.get_default_dtype())

        return weights, ESS

    def _restart_weights(self, ):
        """Restart the weights of the particles."""
        
        return torch.ones(self.params['num_particles']) / self.params['num_particles']

    def _compute_initial_ensemble(self, state_init, pars_init):
        """Compute the initial ensemble."""
        
        state_init_ensemble = state_init
        pars_init_ensemble = pars_init

        '''
        pars_init_ensemble = pars_init.unsqueeze(0).repeat(
            (self.params['num_particles'], 1)
        )
        pdb.set_trace()
        pars_init_ensemble = torch.zeros(
            (self.params['num_particles'], pars_init.shape[0])
        )
        '''
        for i in range(self.params['num_particles']):
            state_init_ensemble[i], pars_init_ensemble[i] = \
                self.forward_model.model_error.get_initial_ensemble(
                    state=state_init_ensemble[i], 
                    pars=pars_init_ensemble[i]
                    )

        return state_init_ensemble, pars_init_ensemble   

    def _compute_prior_particles(self, t_range, state, pars):
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
                        pars=pars[i]
                        )
                state_ensemble.append(particle_state)
                pars_ensemble.append(particle_pars)
                
            state_ensemble = ray.get(state_ensemble)
            pars_ensemble = ray.get(pars_ensemble)

            state_ensemble = np.asarray(state_ensemble)
            pars_ensemble = np.asarray(pars_ensemble)
        else:
            state_ensemble, pars_ensemble, t_vec = compute_prior_particle(
                    forward_model=self.forward_model, 
                    t_range=t_range,
                    state=state, 
                    pars=pars
                )
            state_ensemble = state_ensemble.transpose(1,2)
            #pars_ensemble = pars_ensemble.detach().numpy()

        return state_ensemble, pars_ensemble

    def _compute_likelihood(self, state_ensemble, pars_ensemble, observation, AE):
        """Compute the model likelihood."""
        if False:#ray.is_initialized():
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
            
            with torch.no_grad():
                #state_ensemble = state_ensemble.to('cuda')
                #pars_ensemble = pars_ensemble.to('cuda')
                #AE = AE.to('cuda')
                HF_state = AE.decoder(state_ensemble, pars_ensemble)
                #HF_state = HF_state.to('cpu')
            likelihood = np.zeros(self.params['num_particles'])
            for i in range(self.params['num_particles']):
                likelihood[i] = compute_likelihood(
                    state=HF_state[i], 
                    observation=observation,
                    observation_operator=self.observation_operator,
                    likelihood_params=self.likelihood_params,
                    )

        return likelihood
    

    def _get_resampled_particles(self, state_ensemble, pars_ensemble, weights):
        """Compute the posterior."""
        
        resampled_ids = np.random.multinomial(
            n=self.params['num_particles'],
            pvals=weights.numpy(),
        )
        indeces = np.repeat(
            np.arange(self.params['num_particles']),
            resampled_ids
        )

        return state_ensemble[indeces], pars_ensemble[indeces]
    
    def compute_filtered_solution(
        self,
        true_sol,
        state_init,
        pars_init
    ):
        """Compute the filtered solution."""
        max_seq_len = self.forward_model.time_stepping_model.max_seq_len
        seq_len = 16

        weights = self._restart_weights()
        self.forward_model.model_error.initialize_model_error_distribution(
            state=state_init,
            pars=pars_init
            )

        state_ensemble, pars_ensemble = self._compute_initial_ensemble(
                state_init=state_init,
                pars_init=pars_init
            )
        
        p_bar = tqdm(
            enumerate(zip(true_sol.obs_times_idx[:-1], true_sol.obs_times_idx[1:])),
            total=len(true_sol.obs_times_idx[1:]),
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
            )
            
        state_ensemble_filtered = state_ensemble
        pars_ensemble_filtered = pars_ensemble.unsqueeze(-1).repeat(1, 1, seq_len)
        for i, (t_old, t_new) in p_bar:
            
            state_ensemble_filtered[:, :, -seq_len:] = \
                state_ensemble_filtered[:, :, -seq_len:] + \
                torch.normal(
                    mean=0.,
                    std=self.forward_model.model_error.params['state_std'].item(),
                    size=state_ensemble_filtered[:, :, -seq_len:].shape
                    )
            
            seq_len = t_new - t_old

            self.forward_model.model_error.update_model_error_distribution(
                state=state_ensemble[:, :, -1:],
                pars=pars_ensemble[:, :],
                weights=weights
                )
            # Compute the prior particles
            state_ensemble, pars_ensemble = self._compute_prior_particles(
                t_range=[t_old, t_new],
                state=state_ensemble_filtered[:, :, -max_seq_len:],
                pars=pars_ensemble_filtered[:, :, -1]
            )

            # Compute the likelihood
            likelihood = self._compute_likelihood(
                state_ensemble[:, :, -1],
                pars_ensemble[:, :],
                observation=true_sol.observations[:, :, i+1],
                AE=self.forward_model.AE_model,
                )
            
            weights, ESS = self._update_weights(
                likelihood=likelihood, 
                weights=weights
                )

            if ESS < self.ESS_threshold:
                state_ensemble, pars_ensemble = \
                    self._get_resampled_particles(
                        state_ensemble=state_ensemble,
                        pars_ensemble=pars_ensemble,
                        weights=weights
                    )
                weights = self._restart_weights()

                resample = True
            else:
                resample = False
            
            state_ensemble_filtered = torch.cat(
                [state_ensemble_filtered, state_ensemble], 
                axis=-1
                )
            pars_ensemble_filtered = torch.cat(
                [pars_ensemble_filtered, pars_ensemble.unsqueeze(-1).repeat(1, 1, seq_len)], 
                axis=-1
                )
            
            p_bar.set_postfix({'Resample': resample})

            '''
            HF_state = self.forward_model.AE_model.decoder(
                state_ensemble[:, :, -1], pars_ensemble)
            
            print(likelihood[0:4])
            print(pars_ensemble_filtered[0:4])
            plt.figure()
            plt.plot(HF_state[0, 1, :].detach().numpy(), label='0')
            plt.plot(HF_state[1, 1, :].detach().numpy(), label='1')
            plt.plot(HF_state[2, 1, :].detach().numpy(), label='2')
            plt.plot(HF_state[3, 1, :].detach().numpy(), label='3')
            plt.plot(true_sol.sol[1, :, t_new].detach().numpy(), label='true')
            plt.legend()
            plt.show()
            
            pdb.set_trace()
            '''
            
        return state_ensemble_filtered, pars_ensemble_filtered
    