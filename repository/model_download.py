import logging
import os
from google.cloud import storage
from pyasn1.type.univ import Null


file_name = 'model_molecare_v1.h5'
bucket_name = 'twoay-ltd-molecare.appspot.com' #/ML-Melanoma/model_molecare_v1.h5
project_dir = '/Users/yauhenbichel/DevBox/MoleCare-app/MoleCare-ML'
keyfilepath = '{}/google-cloud/keyfile/{}'.format(project_dir, "service-account.json")
storage_client = storage.Client.from_service_account_json(keyfilepath)
buckets = storage_client.list_buckets()

for bucket in buckets:
    print(bucket.name)

bucket=storage_client.get_bucket(bucket_name)

blobs = bucket.list_blobs()
blob_model = Null 

for blob in blobs:
    print(blob.name)
    if blob.name == "ML-Melanoma/model_molecare_v1.h5" :
        blob_model = blob

folder = '{}/ml-model'.format(project_dir)

if not os.path.exists(folder):
    os.makedirs(folder)

if blob_model != Null :
    destination_uri = '{}/{}'.format(folder, "model_melanoma_v1.h5") 
    blob = bucket.blob(blob_model.name)
    blob.download_to_filename(destination_uri)


