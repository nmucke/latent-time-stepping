from array import array
import os 
from pathlib import Path
import pdb
import numpy as np 
import oci 
from multiprocessing import Process 
from multiprocessing import Semaphore 

# Number of max processes allowed at a time 
concurrency= 5 
sema = Semaphore(concurrency) 
# The root directory path, Replace with your path 
p = Path('/Users/nikolajmucke/cwi/latent-time-stepping/data/training_data/state') 
# The Compartment OCID 
compartment_id = "ocid1.tenancy.oc1..aaaaaaaaeadwopiezanqrr5ybd3w6wwmzzqavceibijgl46upfkgyonu7otq"
# The Bucket name where we will upload 
bucket_name = "bucket-20230222-1753" 

class ObjectStorageClientWrapper:
    def __init__(self, bucket_name):

        config = oci.config.from_file() 
        self.object_storage_client = oci.object_storage.ObjectStorageClient(config)

        self.namespace = object_storage_client.get_namespace().data 

        self.bucket_name = bucket_name

    def put_object(self, destination_path, source_path):
        upload_to_object_storage(
            source_path=source_path,
            destination_path=destination_path,
            object_storage_client=self.object_storage_client,
            namespace=self.namespace
        )

    def get_object(self, source_path, destination_path):
        return download_from_object_storage(
            source_path=source_path,
            destination_path=destination_path,
            object_storage_client=self.object_storage_client,
            namespace=self.namespace
            )


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