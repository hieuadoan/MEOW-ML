import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
import ase.io
import yaml

def load_config():
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_and_preprocess_data():
    config = load_config()
    file_path = config['data']['file_path']
    file_type = config['data']['file_type']
    test_size = config['data']['test_size']
    random_state = config['data']['random_state']
    
    if file_type == 'csv':
        data = pd.read_csv(file_path)
        features = data.drop('target_property', axis=1)
        target = data['target_property']
    elif file_type == 'xyz':
        atoms = ase.io.read(file_path)
        features = atoms.get_all_distances(mic=True)  # Example feature extraction
        target = atoms.info['target_property']
    elif file_type == 'cif':
        structure = CifParser(file_path).get_structures()[0]
        features = structure.distance_matrix  # Example feature extraction
        target = structure.composition.formula
    else:
        raise ValueError("Unsupported file type")

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, features.columns