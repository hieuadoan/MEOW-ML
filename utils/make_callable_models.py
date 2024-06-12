class CallableModel:
    """
    A wrapper class that makes a model callable.

    Parameters:
    model : object
        The underlying model to be wrapped.

    Methods:
    __call__(x)
        Makes the model callable by predicting the output for the given input.
    fit(X, y)
        Fits the underlying model to the given training data.

    """

    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        return self.model.predict(x)
    
    def fit(self, X, y):
        self.model.fit(X, y)