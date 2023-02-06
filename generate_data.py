


import time
import numpy as np
from discontinuous_galerkin.base.base_model import BaseModel
from discontinuous_galerkin.start_up_routines.start_up_1D import StartUp1D
import matplotlib.pyplot as plt
import pdb
import scipy.stats


from matplotlib.animation import FuncAnimation
import ray


class PipeflowEquations(BaseModel):
    """Advection equation model class."""

    def __init__(
        self, **kwargs):
        super().__init__(**kwargs)

        self.L = 2000
        self.d = 0.508
        self.A = np.pi*self.d**2/4
        self.c = 308.
        self.p_amb = 101325.
        self.p_ref = 5016390.
        self.rho_ref = 52.67
        self.e = 1e-8
        self.mu = 1.2e-5

        self.D_orifice = 0.03
        self.A_orifice = np.pi*(self.D_orifice/2)**2

    def update_leak(self, Cd, leak_location):

        self.Cv = self.A/np.sqrt(self.rho_ref/2 * ((self.A/(self.A_orifice*Cd))**2-1))
        self.leak_location = leak_location

    def density_to_pressure(self, rho):
        """Compute the pressure from the density."""

        return self.c**2*(rho - self.rho_ref) + self.p_ref


    def pressure_to_density(self, p):
        """Compute the density from the pressure."""

        return (p - self.p_ref)/self.c**2 + self.rho_ref

    def friction_factor(self, q):
        """Compute the friction factor."""

        rho = q[0]/self.A
        u = q[1]/q[0]

        Re = rho * u * self.d / self.mu
        
        f = (self.e/self.d/3.7)**(1.11) + 6.9/Re
        f *= -1/4*1.8*np.log10(f)**(-2)
        f *= -1/2/self.d * rho * u*u 

        return f

    def system_jacobian(self, q):

        u = q[1]/q[0]

        J =np.array(
            [[0, 1],
            [-u*u + self.c*self.c/np.sqrt(self.A), 2*u]]
            )

        return J

    def initial_condition(self, x):
        """Compute the initial condition."""

        init = np.ones((self.DG_vars.num_states, x.shape[0]))

        init[0] = self.pressure_to_density(self.p_ref) * self.A
        init[1] = init[0] * 4.0

        return init
    
    def boundary_conditions(self, t, q=None):
        """Compute the boundary conditions."""

        rho_out = self.pressure_to_density(self.p_ref)

        BC_state_1 = {
            'left': None,
            'right': rho_out * self.A,
        }
        BC_state_2 = {
            'left': q[0, 0] * 4,#(4.0 + 0.5*np.sin(0.2*t)),
            'right': None
        }

        BCs = [BC_state_1, BC_state_2]
        
        return BCs
    
    def velocity(self, q):
        """Compute the wave speed."""
        
        u = q[1]/q[0]

        c = np.abs(u) + self.c/np.sqrt(self.A)

        return c
        
    def flux(self, q):
        """Compute the flux."""


        p = self.density_to_pressure(q[0]/self.A)

        flux = np.zeros((self.DG_vars.num_states, q.shape[1]))

        flux[0] = q[1]
        flux[1] = q[1]*q[1]/q[0] + p * self.A

        return flux
    
    def source(self, t, q):
        """Compute the source."""

        s = np.zeros((self.DG_vars.num_states,self.DG_vars.Np*self.DG_vars.K))

        point_source = np.zeros((self.DG_vars.Np*self.DG_vars.K))
        if t>0:
            x = self.DG_vars.x.flatten('F')
            width = 10
            point_source = \
                (np.heaviside(x-self.leak_location + width/2, 1) - 
                    np.heaviside(x-self.leak_location-width/2, 1))
            point_source *= 1/width

        rho = q[0]/self.A
        p = self.density_to_pressure(rho)

        s[0] = - self.Cv * np.sqrt(rho * (p - self.p_amb)) * point_source

        s[1] = -self.friction_factor(q)

        return s

@ray.remote(num_returns=1)
def simulate_pipeflow(
    leak_location=100,
    leak_size=1e-4,
    pipe_DG=None,
    idx=0
    ):

    pipe_DG.update_leak(leak_size, leak_location)
    
    init = pipe_DG.initial_condition(pipe_DG.DG_vars.x.flatten('F'))

    t_final = 100.
    sol, t_vec = pipe_DG.solve(t=0, q_init=init, t_final=t_final, print_progress=False)

    x = np.linspace(pipe_DG.DG_vars.x[0, 0], pipe_DG.DG_vars.x[-1, -1], 256)

    u = np.zeros((len(x), len(t_vec)))
    rho = np.zeros((len(x), len(t_vec)))
    for t in range(sol.shape[-1]):
        rho[:, t] = pipe_DG.evaluate_solution(x, sol_nodal=sol[0, :, t])
        u[:, t] = pipe_DG.evaluate_solution(x, sol_nodal=sol[1, :, t])
    rho = rho / pipe_DG.A
    u = u / rho / pipe_DG.A

    rho = rho[:, 1:]
    u = u[:, 1:]

    rho = np.expand_dims(rho, axis=0)
    u = np.expand_dims(u, axis=0)
    state = np.concatenate((rho, u), axis=0)

    pars = np.array([leak_location, leak_size])


    np.save(f'data/raw_data/training_data/pars/sample_{idx}.npy', pars)
    np.save(f'data/raw_data/training_data/state/sample_{idx}.npy', state)

    print('Saved data for sample {}'.format(idx))

    return None



def main():
    xmin = 0.
    xmax = 1000.

    BC_params = {
        'type':'dirichlet',
        'treatment': 'naive',
        'numerical_flux': 'roe',
    }

    steady_state = {
        'newton_params':{
            'solver': 'direct',
            'max_newton_iter': 200,
            'newton_tol': 1e-5
        }
    }

    numerical_flux_type = 'lax_friedrichs'
    numerical_flux_params = {
        'alpha': 0.5,
    }
    '''
    stabilizer_type = 'filter'
    stabilizer_params = {
        'num_modes_to_filter': 10,
        'filter_order': 36,
    }
    '''
    stabilizer_type = 'slope_limiter'
    stabilizer_params = {
        'second_derivative_upper_bound': 1e-4,
    }

    time_integrator_type = 'implicit_euler'
    time_integrator_params = {
        'step_size': 0.05,
        'newton_params':{
            'solver': 'krylov',
            'max_newton_iter': 200,
            'newton_tol': 1e-5
            }
        }

    polynomial_type='legendre'
    num_states=2

    polynomial_order = 3
    num_elements = 125


    pipe_DG = PipeflowEquations(
        xmin=xmin,
        xmax=xmax,
        num_elements=num_elements,
        polynomial_order=polynomial_order,
        polynomial_type=polynomial_type,
        steady_state=steady_state,
        num_states=num_states,
        BC_params=BC_params,
        stabilizer_type=stabilizer_type, 
        stabilizer_params=stabilizer_params,
        time_integrator_type=time_integrator_type,
        time_integrator_params=time_integrator_params, 
        numerical_flux_type=numerical_flux_type,
        numerical_flux_params=numerical_flux_params,
        )


    num_samples = 3000
    leak_location_vec  = np.random.uniform(10, 990, num_samples)
    leak_size_vec = np.random.uniform(1., 2., num_samples)
    #leak_size_vec = scipy.stats.loguniform.rvs(1., 2., size=num_samples)
    
    ray.shutdown()
    ray.init(num_cpus=25)
    ray.get([
        simulate_pipeflow.remote(
            leak_location=leak_location,
            leak_size=leak_size,
            pipe_DG=pipe_DG,
            idx=idx
        ) for (leak_location, leak_size, idx) in 
        zip(leak_location_vec, leak_size_vec, range(num_samples))])

if __name__ == "__main__":
    
    main()
