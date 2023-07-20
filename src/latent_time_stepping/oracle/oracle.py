from array import array
import os 
from pathlib import Path
import pdb
import numpy as np 
import oci 
from multiprocessing import Process 
from multiprocessing import Semaphore 
import ocifs
import torch

# Number of max processes allowed at a time 
concurrency= 5 
sema = Semaphore(concurrency) 
# The root directory path, Replace with your path 
p = Path('/Users/nikolajmucke/cwi/latent-time-stepping/data/training_data/state') 
# The Compartment OCID 
compartment_id = "ocid1.tenancy.oc1..aaaaaaaaeadwopiezanqrr5ybd3w6wwmzzqavceibijgl46upfkgyonu7otq"
# The Bucket name where we will upload 
bucket_name = "bucket-20230222-1753" 

def upload_to_object_storage(
    source_path: str,
    destination_path:str, 
    object_storage_client,
    namespace
): 
    
    with open(source_path, "rb") as in_file: 
        object_storage_client.put_object(namespace,bucket_name,destination_path,in_file) 


def download_from_object_storage(
    source_path,
    destination_path,
    object_storage_client,
    namespace
):
    
    get_obj = object_storage_client.get_object(namespace,bucket_name,source_path,)

    with open(destination_path, 'wb') as f:
        for chunk in get_obj.data.raw.stream(2, decode_content=False):
            f.write(chunk)

class ObjectStorageClientWrapper:
    def __init__(self, bucket_name):

        config = oci.config.from_file() 
        self.object_storage_client = oci.object_storage.ObjectStorageClient(config)

        self.namespace = self.object_storage_client.get_namespace().data 

        self.bucket_name = bucket_name

        self.fs = ocifs.OCIFileSystem(config)

    def put_object(self, data, destination_path): #, source_path):

        with self.fs.open(f'{self.bucket_name}@{self.namespace}/{destination_path}', 'wb') as f:
            if destination_path[-3:] == 'npz':
                np.savez_compressed(f, data=data)
            else:
                np.save(f, data)

        '''
        upload_to_object_storage(
            source_path=source_path,
            destination_path=destination_path,
            object_storage_client=self.object_storage_client,
            namespace=self.namespace
        )
        '''

    def get_object(self, source_path):

        with self.fs.open(f'{self.bucket_name}@{self.namespace}/{source_path}', 'rb') as f:
            data = np.load(f)
            if source_path[-3:] == 'npz':
                data=data['data']

        return data


class OracleDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_path: str,
        num_files: int,
        object_storage_client: ObjectStorageClientWrapper,
        ) -> None:
        super().__init__()

        self.data_path = data_path
        self.num_files = num_files
        self.object_storage_client = object_storage_client

    def __len__(self) -> int:
        return self.num_files

    def __getitem__(self, index: int) -> torch.Tensor:
        state = self.object_storage_client.get_object(
            source_path=f'{self.data_path}/state/sample_{index}.npy'
        )
        pars = self.object_storage_client.get_object(
            source_path=f'{self.data_path}/pars/sample_{index}.npy'
        )

        state = torch.from_numpy(state, dtype=torch.get_default_dtype())
        pars = torch.from_numpy(pars, dtype=torch.get_default_dtype())

        return state, pars


def get_oracle_data(
    bucket_name: str,
    data_path: str,
    num_files: int,
    num_skip_steps: int = 4,
    num_workers: int = 16
) -> torch.Tensor:
    
    batch_size = 100

    object_storage_client = ObjectStorageClientWrapper(bucket_name)

    dataset = OracleDataset(
        data_path=data_path,
        num_files=num_files,
        object_storage_client=object_storage_client,
    )
    dataloaders = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    state, pars = dataset.__getitem__(0)
    state_shape = state.shape, 
    pars_shape = pars.shape

    state = torch.zeros(
        (num_files, state_shape[0], state_shape[1], state_shape[2])
    )
    pars = torch.zeros(
        (num_files, pars_shape[0])
    )

    for i, (state_batch, pars_batch) in enumerate(dataloaders):
        state[i*batch_size:(i+1)*batch_size] = state_batch
        pars[i*batch_size:(i+1)*batch_size] = pars_batch

    return state, pars


if __name__ == '__main__': 
    config = oci.config.from_file() 
    object_storage_client = oci.object_storage.ObjectStorageClient(config) 
    namespace = object_storage_client.get_namespace().data 
    proc_list: array = []

    object_loader = ObjectStorageClientWrapper(bucket_name)

    for i in range(1,5):
        object_loader.put_object(
            destination_path=f'state/sample_{i}.npy',
            source_path=f'/Users/nikolajmucke/cwi/latent-time-stepping/data/training_data/state/sample_{i}.npy'
        )

    for i in range(1,5):
        object_loader.get_object(
            source_path=f'state/sample_{i}.npy',
            destination_path=f'/Users/nikolajmucke/cwi/latent-time-stepping/sample_{i}.npy'
        )