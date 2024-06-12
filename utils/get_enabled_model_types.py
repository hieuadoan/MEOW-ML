def get_enabled_model_types(config):
    model_types = []
    if config['models']['traditional']['linear_regression']['enabled']:
        model_types.append('linear_regression')
    if config['models']['traditional']['random_forest']['enabled']:
        model_types.append('random_forest')
    if config['models']['traditional']['svr']['enabled']:
        model_types.append('svr')
    if config['models']['neural_network']['enabled']:
        model_types.append('neural_network')
    if config['models']['gnn']['enabled']:
        model_types.append('gnn')
    return model_types