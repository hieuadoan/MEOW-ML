import os

def get_dataset_name(config):
    file_path = config['data']['file_path']
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    
    return dataset_name