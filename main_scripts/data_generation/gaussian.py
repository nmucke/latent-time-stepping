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

class Gaussian():
    def __init__(
        self,
        N: int = 90,
        a: float = 8,
        a0 = 0.1,
        a1 = 0.1,
        b = 0.1,
    ) -> None:
        self.N = N
        self.a = a
        self.a0 = a0
        self.a1 = a1
        self.b = b

        self.X = np.arange(0, int(np.sqrt(self.N)))
        self.X = np.array(np.meshgrid(self.X, self.X)).T.reshape(-1, 2)

        self.x0 = np.zeros((int(np.sqrt(self.N)), int(np.sqrt(self.N))))

        self.covariance_matrix = np.zeros((self.N, self.N))

        for i in range(self.N):
            for j in range(self.N):
                self.covariance_matrix[i, j] = self.a0*np.exp(-np.linalg.norm(self.X[i] - self.X[j])**2/self.b)

                if i == j:
                    self.covariance_matrix[i, j] += self.a1
        
    def solve(
        self, 
        t_steps: int = 10,
    ):
        
        x = []
        x_old = self.x0.flatten()

        for i in range(t_steps):

            noise = np.random.multivariate_normal(self.x0.flatten(), self.covariance_matrix)

            x_new = self.a * x_old + noise

            x.append(x_new)

            x_old = x_new   


        return np.array(x)
    

#@ray.remote(num_cpus=1)
def simulate_gaussian(
    t_steps, 
    a, 
    idx, 
    to_oracle, 
    train_or_test,
):

    gaussian = Gaussian(
        N=64,
        a=a,
        a0 = 3.0,
        a1 = 0.01,
        b = 20,
    )

    sol = gaussian.solve(
        t_steps=t_steps,
    )

    sol = sol.transpose()
    sol = np.expand_dims(sol, 0)

    if to_oracle:
        path = f'gaussian_phase/raw_data/{train_or_test}'
    else:
        path = f'data/gaussian_phase/raw_data/{train_or_test}'


    save_data(
        idx=idx,
        pars=np.array([a]),
        state=sol,
        path=path,
        to_oracle=to_oracle
    )

    return None

def main():


    TRAIN_OR_TEST = 'train'

    # Create directory if it does not exist
    #create_directory(save_string)

    a_list = np.random.uniform(low=0.9, high=0.9, size=3000)

    remote_list = []
    for idx, a in enumerate(a_list):
        #remote_list.append(simulate_lorenz.remote(
        remote_list.append(simulate_gaussian(
                t_steps=10, 
                a=a, 
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