
import os
import ray
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import pdb
from latent_time_stepping.oracle import ObjectStorageClientWrapper
from latent_time_stepping.utils import create_directory

def save_data(
    idx: int, 
    pars: np.ndarray, 
    state: np.ndarray,
    path: str, 
    to_oracle: bool =False
    ):

    pars_path = f'{path}/pars'
    state_path = f'{path}/state'

    if to_oracle:

        bucket_name = "bucket-20230222-1753"

        # upload to oracle
        object_storage_client = ObjectStorageClientWrapper(bucket_name)

        object_storage_client.put_numpy_object(
            data=state,
            destination_path=f'{state_path}/sample_{idx}.npz',
        )

        object_storage_client.put_numpy_object(
            data=pars,
            destination_path=f'{pars_path}/sample_{idx}.npz',
        )

    else:

        # Create directory if it does not exist
        if not os.path.exists(pars_path):
            os.makedirs(pars_path)
        if not os.path.exists(state_path):
            os.makedirs(state_path)

        np.savez_compressed(f'{pars_path}/sample_{idx}', data=pars)
        np.savez_compressed(f'{state_path}/sample_{idx}', data=state)

    print('Saved data for sample {}'.format(idx))

    return None

class Burgers():
    def __init__(
        self,
        N: int = 256,
        u: float = 2,
        num_steps: int = 300,
    ) -> None:
        self.N = N
        self.u = u

        self.nu = 1/150

        time_step_size = 0.001
        self.t = np.arange(0, time_step_size*num_steps, time_step_size)

        self.x = np.linspace(0, 2, N)

        self.init = u*np.sin(2*np.pi*self.x/2)


        # Second derivative finite difference matrix
        self.second_deriv_matrix = -2*np.eye(self.N-2)
        self.second_deriv_matrix += np.diag(np.ones(self.N-2-1), k=1)
        self.second_deriv_matrix += np.diag(np.ones(self.N-2-1), k=-1)
        self.second_deriv_matrix = np.concatenate(
            (
                np.zeros((self.N-2, 1)),
                self.second_deriv_matrix,
                np.zeros((self.N-2, 1)),
            ),  
            axis=1     
        ) 
        self.second_deriv_matrix[0, 0] = 1
        self.second_deriv_matrix[-1, -1] = 1
        self.second_deriv_matrix /= (self.x[1] - self.x[0])**2

        self.first_derivative_matrix = np.diag(np.ones(self.N-2-1), k=1)
        self.first_derivative_matrix -= np.diag(np.ones(self.N-2-1), k=-1)
        self.first_derivative_matrix = np.concatenate(
            (
                np.zeros((self.N-2, 1)),
                self.first_derivative_matrix,
                np.zeros((self.N-2, 1)),
            ),          
            axis=1
        )
        self.first_derivative_matrix[0, 0] = -1
        self.first_derivative_matrix[-1, -1] = 1
        self.first_derivative_matrix /= 4*(self.x[1] - self.x[0])
    
    def rhs(self, q, t):
        """ Space discretization of Burgers equations """

        dudt = - self.first_derivative_matrix @ (q*q) + self.nu*self.second_deriv_matrix @ q

        # Add zero boundary conditions
        dudt = np.concatenate(
            (
                np.zeros(1),
                dudt,
                np.zeros(1),
            )
        )
        return dudt
    
    def solve(
        self, 
    ):

        sol = odeint(
            self.rhs, 
            self.init, 
            self.t,
        )

        return sol
    

#@ray.remote(num_cpus=1)
def simulate_burgers(
    N: int = 256,
    u: float = 2,
    num_steps: int = 300,
    train_or_test: str = 'train',
    idx: int = 0
):
    
    burgers = Burgers(
        N=N,
        u=u,
        num_steps=num_steps,
    )

    sol = burgers.solve()

    sol = sol.transpose()
    sol = np.expand_dims(sol, 0)

    path = f'data/burgers_phase/raw_data/{train_or_test}'


    save_data(
        idx=idx,
        pars=np.array([u]),
        state=sol,
        path=path,
        to_oracle=False
    )

    return None

def main():


    TRAIN_OR_TEST = 'test'

    # Create directory if it does not exist
    #create_directory(save_string)

    u_list = np.random.uniform(low=0.5, high=1.5, size=20)

    remote_list = []
    for idx, u in enumerate(u_list):
        #remote_list.append(simulate_lorenz.remote(
        remote_list.append(simulate_burgers(
                N=256, 
                u=u, 
                num_steps=300 if TRAIN_OR_TEST=='train' else 600,
                idx=idx, 
                train_or_test=TRAIN_OR_TEST,
            )
        )

    ray.get(remote_list)

if __name__ == "__main__":
    ray.init(num_cpus=25)
    main()
    ray.shutdown()



'''
# Plot the first three variables
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot(x[:, 0], x[:, 1], x[:, 2])
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")
plt.show()
'''