from google.cloud import storage

# Initialise a client
storage_client = storage.Client("[twoay-ltd-molecare.appspot.com]")
#storage_client.
# Create a bucket object for our bucket
bucket = storage_client.get_bucket("bucket_name")
# Create a blob object from the filepath
#blob = bucket.blob("folder_one/foldertwo/filename.extension")
# Download the file to a destination
#blob.download_to_filename(destination_file_name)