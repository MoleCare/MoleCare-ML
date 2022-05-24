import os

from google.oauth2 import service_account
import googleapiclient.discovery

# sa-name is the name of your service account, 
# and project-id is the ID of your Google Cloud project

def create_key(service_account_email):
    """Creates a key for a service account."""

    credentials = service_account.Credentials.from_service_account_file(
        filename=os.environ['GOOGLE_APPLICATION_CREDENTIALS'],
        scopes=['https://www.googleapis.com/auth/cloud-platform'])

    service = googleapiclient.discovery.build(
        'iam', 'v1', credentials=credentials)

    key = service.projects().serviceAccounts().keys().create(
        name='projects/-/serviceAccounts/' + service_account_email, body={}
        ).execute()

    print('Created key: ' + key['name'])

