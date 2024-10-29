from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
import os


# Authenticate using service account credentials
gauth = GoogleAuth()
scope = ['https://www.googleapis.com/auth/drive']
gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name('service_account_key.json', scope)
drive = GoogleDrive(gauth)

def download_folder(folder_id, local_path):
    """Download all contents of a Google Drive folder recursively."""
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    # Get list of all files and folders in the specified folder
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

    for file in file_list:
        file_name = file['title']
        file_id = file['id']

        if file['mimeType'] == 'application/vnd.google-apps.folder':  # Check if it's a folder
            # Recursive call for subfolder
            print(f'Downloading folder: {file_name}')
            download_folder(file_id, os.path.join(local_path, file_name))
        else:
            # Download file
            print(f'Downloading file: {file_name}')
            file.GetContentFile(os.path.join(local_path, file_name))

# Replace with the Google Drive folder ID you want to download and the local path to save to
root_folder_id = '1-iahx_9D5aGMsUHwMkMcFOuB_dhG9ZYM'
local_path = '/home/mburu/Master_Thesis/master-thesis-da/datasets/'
download_folder(root_folder_id, local_path)