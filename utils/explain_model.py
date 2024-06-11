import shap
import torch

def explain_model(model, X_train, X_test, feature_names):
    if isinstance(model, torch.nn.Module):
        explainer = shap.DeepExplainer(model, torch.tensor(X_train, dtype=torch.float32))
        shap_values = explainer.shap_values(torch.tensor(X_test, dtype=torch.float32))
    else:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type='dot')