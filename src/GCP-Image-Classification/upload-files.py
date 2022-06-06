from google.cloud import storage
import os
from os.path import exists

os.environ.setdefault("GCLOUD_PROJECT", "tactile-rigging-352016")

bucket_name = "YOUR-GCS-BUCKET"
destination_blob_name_mel = "dataset_derm/Melanoma"
destination_blob_name_not_mel = "dataset_derm/NotMelanoma"

melanoma_source_folder = "/Users/yauhenbichel/Downloads/DermMel/DermMel/Melanoma"
notmelanoma_source_folder = "/Users/yauhenbichel/Downloads/DermMel/DermMel/NotMelanoma"

for (dirpath, dirnames, filenames)  in os.walk(melanoma_source_folder):
    for file in filenames:
        file = str(dirpath + "/" + file)
        print(file)
        if exists(file):
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(destination_blob_name_mel)
            blob.upload_from_filename(file)

for (dirpath, dirnames, filenames) in os.walk(notmelanoma_source_folder):
    for file in filenames:
        file = str(dirpath + "/" + file)
        print(file)
        if exists(file):
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(destination_blob_name_not_mel)
            blob.upload_from_filename(file)