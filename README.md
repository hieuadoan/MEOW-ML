<!-- Include logo -->
<p align="center">
<img src="https://github.com/hieuadoan/MEOW-ML/blob/main/MEOW-ML.png" alt="Project Logo" width="400"/>
</p>

**MEOW-ML** (Materials Evaluation and Optimization Workflow with Machine Learning) is a playful yet powerful automated machine learning (AutoML) pipeline designed for materials science research. This project leverages traditional machine learning models, deep learning models, and Graph Neural Networks (GNNs) to provide an end-to-end solution for predicting material properties from various data formats, including `.csv`, `.xyz`, and `.cif` files.

## Features

- **Automated Data Preprocessing**: Supports preprocessing for `.csv`, `.xyz`, and `.cif` files, ensuring seamless integration with materials science datasets.
- **Model Diversity**: Includes traditional ML models (Linear Regression, Random Forest, SVR), deep learning models (feedforward neural networks), and advanced Graph Neural Networks (GNNs).
- **Hyperparameter Tuning**: Utilizes Optuna for efficient hyperparameter optimization, ensuring the best model performance.
- **Model Explainability**: Integrates SHAP for explaining model predictions, providing insights into feature importance and model behavior.
- **Extensible Architecture**: Modular design allows for easy addition of new models, data preprocessing steps, and explainability techniques.

## Installation

To create the environment and install the dependencies, run:

```sh
conda env create -f environment.yml
```
<u>Note</u>: Replace *environment.yml* with *environment-gpu.yml* if you can run CUDA-enabled PyTorch.

Then, activate the environment:

```sh
conda activate meow-ml
```

Get site-packages directory
```sh
SP_DIR=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
```

Create the .path file to set the project folder
```sh
echo "/path/to/your/project" > "$SP_DIR/meow-ml.pth"
```

Replace the /path/to/your/project with the path to the cloned MEOW-ML directory.

## Usage
1. **Prepare data**: Ensure data is in one of the supported formats (.csv, .xyz, .cif) and update the configs/config.yaml file with the appropriate file path and settings.

2. **Run hyperparameter tuning**: Optimize hyperparameters for the best model performance.

```sh
python hyperparameter_tuning/tuner.py
```

3. **Train and evaluate models**: Train and evaluate the models using the best hyperparameters.

```sh
python main.py
```

4. **Explain model predictions**: Generate explanations for model predictions to gain insights into feature importance.

```
python utils/explain_model.py
```

## Project structure
```sh
MEOW-ML/
│
├── configs/
│   └── config.yaml
├── data/
│   ├── __init__.py
│   ├── data_preparation.py
│   └── datasets.py
├── models/
│   ├── __init__.py
│   ├── model_definitions.py
│   └── train_evaluate.py
├── gnn_models/
│   ├── __init__.py
│   ├── gnn_definitions.py
│   └── train_gnn.py
├── hyperparameter_tuning/
│   ├── __init__.py
│   └── tuner.py
├── utils/
│   ├── __init__.py
│   ├── load_config.py
│   ├── make_callable_models.py
│   └── explain_model.py
├── main.py
├── environment.yml
├── environment-gpu.yml
└── README.md
```

## Acknowledgements
- This project uses Scikit-learn, Pytorch, Pytorch Geometric, Optuna, ASE, Pymatgen and SHAP.
