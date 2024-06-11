import yaml

def load_config(configPath='configs/config.yaml'):
    with open(configPath, 'r') as file:
        config = yaml.safe_load(file)
    return config