import pdb
import torch
import torch.nn as nn

class Preprocessor(nn.Module):
    def __init__(self, num_states, num_pars):

        self.state_min = [1e12 for _ in range(num_states)]
        self.state_max = [-1e12 for _ in range(num_states)]

        self.pars_min = [1e12 for _ in range(num_pars)]
        self.pars_max = [-1e12 for _ in range(num_pars)]

    def partial_fit_state(self, state):

        for i in range(state.shape[1]):
            self.state_min[i] = min(self.state_min[i], state[:, i].min())
            self.state_max[i] = max(self.state_max[i], state[:, i].max())
    
    def partial_fit_pars(self, pars):

        for i in range(pars.shape[1]):
            self.pars_min[i] = min(self.pars_min[i], pars[:, i].min())
            self.pars_max[i] = max(self.pars_max[i], pars[:, i].max())

    def transform_state(self, state, ensemble=False):
        """Transform the state to be between 0 and 1."""

        if ensemble:
            for i in range(state.shape[1]):
                state[:, i] = (state[:, i] - self.state_min[i])/(self.state_max[i] - self.state_min[i])
        else:
            for i in range(state.shape[0]):
                state[i] = (state[i] - self.state_min[i])/(self.state_max[i] - self.state_min[i])
        
        return state

    def inverse_transform_state(self, state, ensemble=False):
        """Inverse transform the state"""

        if ensemble:
            for i in range(state.shape[1]):
                state[:, i] = state[:, i]*(self.state_max[i] - self.state_min[i]) + self.state_min[i]
        else:
            for i in range(state.shape[0]):
                state[i] = state[i]*(self.state_max[i] - self.state_min[i]) + self.state_min[i]

        return state

    def transform_pars(self, pars, ensemble=False):
        """Transform the parameters to be between 0 and 1."""

        if ensemble:
            for i in range(pars.shape[1]):
                pars[:, i] = (pars[:, i] - self.pars_min[i])/(self.pars_max[i] - self.pars_min[i])
        else:
            for i in range(pars.shape[0]):
                pars[i] = (pars[i] - self.pars_min[i])/(self.pars_max[i] - self.pars_min[i])

        return pars

    def inverse_transform_pars(self, pars, ensemble=False):
        """Inverse transform the parameters"""

        if ensemble:
            for i in range(pars.shape[1]):
                pars[:, i] = pars[:, i]*(self.pars_max[i] - self.pars_min[i]) + self.pars_min[i]
        else:
            for i in range(pars.shape[0]):
                pars[i] = pars[i]*(self.pars_max[i] - self.pars_min[i]) + self.pars_min[i]

        return pars

