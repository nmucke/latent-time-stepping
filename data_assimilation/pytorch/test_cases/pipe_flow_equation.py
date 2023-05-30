
import numpy as np
import torch
from data_assimilation.pytorch.forward_model import BaseForwardModel
from data_assimilation.pytorch.model_error import BaseModelError
from data_assimilation.pytorch.observation_operator import BaseObservationOperator
import pdb
import matplotlib.pyplot as plt
import torch.nn as nn

class ModelError(BaseModelError):
    def __init__(
        self, 
        params: dict, 
        latent_state_dims=None,
        full_state_dims=None, 
        pars_dims=None
        ):
        
        self.params = params
        self.latent_state_dims = latent_state_dims
        self.full_state_dims = full_state_dims
        self.pars_dims = pars_dims

        self.smoothing_factor = self.params['smoothing_factor']
        self.a = np.sqrt(1 - self.smoothing_factor)

    def get_initial_ensemble(self, state, pars):
        """Get the initial model error."""

        '''
        pars[0] = np.random.uniform(
            low=0,
            high=1.,
            )
        pars[1] = np.random.uniform(
            low=0.,
            high=1.,
            )
        '''

        state_noise = torch.normal(
            mean=0.,#torch.Tensor([0.]),
            std=self.params['state_std'].item(),
            size=state.shape
            )

        state = state + state_noise
        
        return state, pars

    def initialize_model_error_distribution(self, state, pars):
        """Initialize the model error."""

        self.pars_weighted_mean = pars
        self.pars_covariance = torch.diag(self.params['pars_std'])

    def update_model_error_distribution(self, state, pars, weights):
        """Update the model error distribution."""
        
        self.pars_weighted_mean = torch.matmul(pars.T, weights)

        pars_mean_shifted = pars - self.pars_weighted_mean

        self.pars_covariance = torch.matmul(
            pars_mean_shifted.T, 
            torch.matmul(torch.diag(weights), pars_mean_shifted)
            )
        #self.pars_covariance = torch.diag(self.params['pars_std'])

        self.params['pars_std'] = 0.7 * self.params['pars_std']
        '''
        self.pars_weighted_mean = torch.zeros(self.pars_dims)
        self.pars_var = torch.zeros(self.pars_dims)
        for i in range(self.pars_dims):
            self.pars_weighted_mean[i] = torch.matmul(pars[:, i].T, weights)

            pars_mean_shifted = pars[:, i] - self.pars_weighted_mean[i]

            self.pars_var[i] = torch.matmul(
                pars_mean_shifted.T,
                torch.matmul(torch.diag(weights), pars_mean_shifted)
                )
        '''
    def _sample_noise(self, state, pars):
        """Sample noise."""
        state_noise = torch.normal(
            mean=0.,
            std=self.params['state_std'].item(),
            size=state.shape
            )
            
        pars_mean = self.a*pars + (1 - self.a)*self.pars_weighted_mean
        
        pars_noise = np.random.multivariate_normal(
            mean=pars_mean.numpy(),#pars.numpy(),#
            cov=self.smoothing_factor[0]*self.pars_covariance,
            )
        pars_noise = torch.tensor(pars_noise, dtype=torch.float32)

        '''
        r1 = pars - self.params['pars_std']#self.smoothing_factor*self.pars_covariance.diag()#
        r2 = pars + self.params['pars_std']#self.smoothing_factor*self.pars_covariance.diag()#self.params['pars_std']
        pars_noise = (r1 - r2) * torch.rand(pars.shape) + r2
        pars_noise[pars_noise<0] = 1e-8
        pars_noise[pars_noise>1] = 1 - 1e-8
        '''

        '''
        pars_noise = torch.normal(
            mean=pars_mean,
            std=self.smoothing_factor*self.pars_var,
            )
        '''
        return state_noise, pars_noise

    def get_model_error(self, state, pars):
        """Compute the model error."""

        state_noise, pars_noise = self._sample_noise(state, pars)

        return state_noise, pars_noise
    
    def add_model_error(self, state, pars):
        """Add the model error to the state and parameters."""
        state_noise, pars_noise = self._sample_noise(state, pars)

        state = state + state_noise
        pars = pars_noise

        return state, pars

class PipeFlowForwardModel(BaseForwardModel):
    def __init__(
        self, 
        time_stepping_model: nn.Module,
        AE_model: nn.Module,
        model_error_params: dict =None,
        ):
        super().__init__()

        self.time_stepping_model = time_stepping_model
        self.AE_model = AE_model

        if model_error_params is not None:
            self.model_error_params = model_error_params
            self.model_error = ModelError(
                self.model_error_params,
                latent_state_dims=(self.time_stepping_model.latent_dim,),
                full_state_dims=(2, 256),
                pars_dims=2,
                )

    def update_params(self, pars): 
        self.pars = pars
        
    def compute_forward_model(self, t_range, state, pars):
        """Compute the forward model."""

        output_seq_len = t_range[-1] - t_range[0]

        state = state.transpose(1, 2)

        with torch.no_grad():
            latent_state = self.time_stepping_model.multistep_prediction(
                state,
                pars,
                output_seq_len=output_seq_len,
                )
        
        t_vec = 0

        return latent_state[:, -output_seq_len:], t_vec


class ObservationOperator(BaseObservationOperator):
    def __init__(self, params):
        
        self.params = params

    def get_observations(self, state):
        """Compute the observations."""
        
        if len(state.shape) == 4:
            return state[-1:, :, self.params['observation_index']]
        else:
            return state[-1:, self.params['observation_index']]


class TrueSolution():
    """True solution class."""

    def __init__(
        self, 
        x_vec: torch.tensor,
        t_vec: torch.tensor,
        sol: torch.tensor,
        pars: torch.tensor,
        observation_operator: BaseObservationOperator,
        obs_times_idx: torch.tensor,
        observation_noise_params: dict,
        ) -> None:

        self.sol = sol
        self.pars = pars

        self.x_vec = x_vec 
        self.t_vec = t_vec

        self.obs_times_idx = obs_times_idx
        self.observation_operator = observation_operator
        self.observation_noise_params = observation_noise_params

        self.observations = self.get_observations(with_noise=True)
        self.obs_x = self.x_vec[self.observation_operator.params['observation_index']]
        self.obs_t = self.t_vec[obs_times_idx]
    
    def get_observations(self, with_noise=True):
        """Compute the observations."""
    
        observations = self.observation_operator.get_observations(
            self.sol[:, :, self.obs_times_idx]
            )


        if not with_noise:
            return observations
        else:
            observations_noise = np.random.normal(
                loc=0.,
                scale=self.observation_noise_params['std'],
                size=observations.shape,
                )

            return observations + observations_noise


