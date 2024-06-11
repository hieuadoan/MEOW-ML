class CallableModel:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        return self.model.predict(x)
    
    def fit(self, X, y):
        self.model.fit(X, y)