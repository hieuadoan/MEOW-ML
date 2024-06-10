![alt text](https://github.com/hieuadoan/MEOW-ML/blob/main/MEOW-ML.png?raw=true)
# MEOW-ML
MEOW-ML is a playful yet powerful automated machine learning (AutoML) pipeline designed for materials science research. This project leverages traditional machine learning models, deep learning models, and Graph Neural Networks (GNNs) to provide an end-to-end solution for predicting material properties from various data formats, including `.csv`, `.xyz`, and `.cif` files.

## Features

- **Automated Data Preprocessing**: Supports preprocessing for `.csv`, `.xyz`, and `.cif` files, ensuring seamless integration with materials science datasets.
- **Model Diversity**: Includes traditional ML models (Linear Regression, Random Forest, SVR), deep learning models (feedforward neural networks), and advanced Graph Neural Networks (GCNs).
- **Hyperparameter Tuning**: Utilizes Optuna for efficient hyperparameter optimization, ensuring the best model performance.
- **Model Explainability**: Integrates SHAP for explaining model predictions, providing insights into feature importance and model behavior.
- **Extensible Architecture**: Modular design allows for easy addition of new models, data preprocessing steps, and explainability techniques.

## Installation

To install the required dependencies, run:

```sh
pip install -r requirements.txt
