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

class Lorenz96():
    def __init__(
        self,
        N: int = 90,
        F: float = 8,
    ) -> None:
        self.N = N
        self.F = F   
         

    def L96(self, x, t):
        """Lorenz 96 model with constant forcing"""
        # Setting up vector
        d = np.zeros(self.N)
        # Loops over indices (with operations and Python underflow indexing handling edge cases)
        for i in range(self.N):
            d[i] = (x[(i + 1) % self.N] - x[i - 2]) * x[i - 1] - x[i] + self.F

        return d
    
    def solve(
        self, 
        t_final: float = 30.0,
        step_size: float = 0.01
    ):
        
        x0 = self.F * np.ones(self.N)  # Initial state (equilibrium)
        #x0 = self.F*np.sin(np.linspace(0, 2*np.pi, self.N))# + perturbation
        #x0[0] += 0.1  # Add small perturbation to the first variable

        # add noise to initial state
        #x0 += np.random.normal(0, 0.01, self.N)

        # add noise to a random component
        x0[np.random.randint(0, self.N)] += np.random.normal(0, 0.01)

        t = np.arange(0.0, t_final, step_size)

        x = odeint(self.L96, x0, t)

        return x
    

#@ray.remote(num_cpus=1)
def simulate_lorenz(
    t_final, 
    step_size, 
    F, 
    idx, 
    to_oracle, 
    train_or_test,
):

    lorenz = Lorenz96(
        N=40,
        F=F
    )

    sol = lorenz.solve(
        t_final=t_final,
        step_size=step_size
    )

    sol = sol.transpose()
    sol = np.expand_dims(sol, 0)
    sol = sol[:, :, 100:]

    if to_oracle:
        path = f'lorenz_phase/raw_data/{train_or_test}'
    else:
        path = f'data/lorenz_phase/raw_data/{train_or_test}'


    save_data(
        idx=idx,
        pars=np.array([F]),
        state=sol,
        path=path,
        to_oracle=to_oracle
    )

    return None

def main():

    TRAIN_OR_TEST = 'train'

    # Create directory if it does not exist
    #create_directory(save_string)

    F_list = np.random.uniform(low=8, high=8, size=3000)

    remote_list = []
    for idx, F in enumerate(F_list):
        #remote_list.append(simulate_lorenz.remote(
        remote_list.append(simulate_lorenz(
                t_final=40, 
                step_size=0.05, 
                F=F, 
                idx=idx, 
                to_oracle=False, 
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