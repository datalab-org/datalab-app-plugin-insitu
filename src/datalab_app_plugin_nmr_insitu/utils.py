import os
from datalab_api import DatalabClient
from zipfile import ZipFile


DATALAB_API_URL = "https://demo-api.datalab-org.io"
client = DatalabClient(DATALAB_API_URL)


def check_file_exist(item_id, file_path):
    item = client.get_item(item_id)
    files = item.get('files', [])
    file_exists = any(file['original_name'] ==
                      os.path.basename(file_path) for file in files)
    return file_exists

    # if check_file_exist(item_id, file_path):
    #     return
    # else:


def get_exp_data(item_id):
    data_zip = client.get_item_files(item_id)

    if not data_zip:
        print(f"Failed to retrieve the zip file for item ID: {item_id}")
        return None

    os.makedirs('tmp_data', exist_ok=True)

    try:
        with ZipFile(data_zip, 'r') as data_folder:
            data_folder.extractall('tmp_data')
    except Exception as e:
        print(f"Error during extraction: {e}")
        return None

    return 'tmp_data'


def save_data(item_id, file_path):
    client.upload_file(file_path=file_path, item_id=item_id)
