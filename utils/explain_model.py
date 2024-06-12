import shap
import torch

def explain_model(model, X_train, X_test, feature_names):
    """
    Explains the model's predictions using SHAP (SHapley Additive exPlanations).

    Parameters:
        model (object): The machine learning model to be explained.
        X_train (array-like): The training data used to train the model.
        X_test (array-like): The test data for which predictions are to be explained.
        feature_names (list): List of feature names corresponding to the columns of X_train and X_test.

    Returns:
        None
    """
    if isinstance(model, torch.nn.Module):
        explainer = shap.DeepExplainer(model, torch.tensor(X_train, dtype=torch.float32))
        shap_values = explainer.shap_values(torch.tensor(X_test, dtype=torch.float32))
    else:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type='dot')